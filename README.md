# MITRE RAG SOC Assistant

Petit prototype qui enrichit des alertes avec le contexte MITRE ATT&CK (via embeddings + FAISS) puis envoie ces informations à un LLM (OpenRouter ou un modèle local Hugging Face) pour produire une analyse SOC détaillée.

- Interface graphique : `web_app.py` (FastAPI + `templates/index.html`)

---

## Prérequis
- Python 3.10+ recommandé
- Accès réseau pour télécharger le JSON MITRE (ou conservez `rag_mitre_openrouter/enterprise-attack.json` localement)

---

## Installation
1. Créez et activez un environnement virtuel :

```powershell
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
```

2. Installez les dépendances :

```bash
pip install fastapi uvicorn[standard] sentence-transformers faiss-cpu openai python-dotenv requests jinja2 transformers accelerate
```

3. Créez un fichier `.env` (voir l'exemple ci‑dessous).

---

## Exemple de `.env`
```
OPENROUTER_API_KEY=sk-or-...
# Optionnel : utiliser un modèle local
# USE_LOCAL_LLM=true
# LOCAL_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
```

- Si `USE_LOCAL_LLM=true`, aucune clé OpenRouter n'est requise et l'application tentera de charger `LOCAL_LLM_MODEL` localement.

---

## Lancer l'application

```bash
uvicorn web_app:app --reload --host 127.0.0.1 --port 8811
```

Puis ouvrez : http://127.0.0.1:8811/

---

