# Lumina

A Gen-Z mental wellness chat companion. Not a therapist, not a replacement for professional care or emergency services.

Per message: **rule-based safety gate** + **fine-tuned multi-task Transformer** (emotion, intensity, risk) → **safety decision** →
CRITICAL: fixed crisis message, everything else skipped · otherwise: **ChromaDB memory** of past session notes → **Gemini** reply.

## Run it (Windows, from this folder)

```powershell
copy .env.example .env                  # then paste your GEMINI_API_KEY into .env
.venv\Scripts\activate                  # first time: python -m venv .venv; pip install -r requirements.txt
cd frontend; npm install; npm run build; cd ..
uvicorn backend.app:app                 # open http://localhost:8000
```

Frontend hot-reload while editing: `cd frontend; npm run dev` (http://localhost:5173, proxies `/api` to :8000).

Tests: `python -m pytest`

## Train the model

CSV columns: `text, emotion, intensity (0-1), risk (safe|caution|critical)`, optional `split (train|val|test)`.
See `data/example.csv` for the format (30 synthetic rows, only good for checking the pipeline runs).

```powershell
python train.py --data data/your_dataset.csv            # distilbert-base-uncased by default
python train.py --data data/your_dataset.csv --model roberta-base --epochs 5
```

Saves the best epoch (chosen on validation: critical recall first, then macro F1) to `models/lumina/`,
scores the untouched test set once, and writes `models/lumina/metrics.json`. Restart the backend to load it.
No model folder = the app runs on the rule gate alone, and the UI says so.

No GPU? Run `train.py` on Google Colab (GPU runtime), then copy the `models/lumina/` folder back here.

## Where things are

| File | What it does |
|---|---|
| `backend/safety.py` | Rule gate (phrases, leetspeak, elongation, misspellings, denials), `decide()`, fixed crisis text |
| `backend/ml.py` | Multi-task model (shared encoder + 3 heads), shared by training and inference |
| `train.py` | Fine-tuning, model selection, test metrics |
| `backend/memory.py` | ChromaDB store of de-identified session notes |
| `backend/llm.py` | Gemini therapist reply + de-identified summaries, retry/backoff, fallback |
| `backend/app.py` | FastAPI pipeline + SQLite sessions; serves `frontend/dist` |
| `frontend/src/App.jsx` | Chat, past sessions, breathing exercise, "Under the hood" pipeline view |

Privacy: when a session ends, its transcript is deleted. Only a de-identified note (for memory) and mood stats remain, locally.
HIPAA-inspired data minimization principles, not HIPAA compliance.
