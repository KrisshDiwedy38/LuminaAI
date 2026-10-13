# Lumina

A Gen-Z mental wellness chat companion. Not a therapist, not a replacement for professional care or emergency services.

Per message: **rule-based safety gate** + **fine-tuned multi-task Transformer** (emotion, intensity, risk) → **safety decision** →
CRITICAL: fixed crisis card with verified helplines, everything else skipped · otherwise: **ChromaDB memory** of past session notes → **Gemini** reply.

## Run it (Windows, from this folder)

```powershell
copy .env.example .env                  # then paste your GEMINI_API_KEY into .env (free: aistudio.google.com/apikey)
.venv\Scripts\activate                  # first time: python -m venv .venv; pip install -r requirements.txt
cd frontend; npm install; npm run build; cd ..
uvicorn backend.app:app                 # opens http://localhost:8000 in your browser
```

After changing anything in `frontend/`, run `npm run build` again and refresh. Hot reload while editing:
`cd frontend; npm run dev` (http://localhost:5173, proxies `/api` to :8000).

Tests: `python -m pytest` (no Gemini calls, safe to run any time).

Model choice lives in `.env`: `GEMINI_MODEL` (default `gemini-3.5-flash-lite`) and `GEMINI_FALLBACK_MODEL`
(default `gemini-3.5-flash`, used when the first is rate-limited).

## What's in the app

- **Chat** with a calm companion; optional Gen-Z tone; off-topic requests are gently redirected
- **Crisis card**: fixed text with tap-to-call India helplines (112, Tele-MANAS 14416 / 1-800-891-4416)
- **Past sessions**: ending a session saves a de-identified note (used as memory) and deletes the transcript
- **Under the hood**: shows each reply's 5 pipeline steps (rules, ML, decision, memory, reply)
- **Breathing exercise**: 4-4-6 guided breathing
- Light and dark themes, desktop / tablet / mobile layouts

## Safety behaviour

- Phrase rules handle leetspeak (`k1ll`), elongation (`kiiill`), misspellings (`sucide`) and method questions
- Context: a vague wish ("i wanna feel that") soon after talk of death is treated as a crisis
- Denials ("i dont wanna die") get a gentle check-in, not the crisis card
- The next reply after a crisis first checks the person is okay to continue
- Gemini never supplies phone numbers or sites; any it writes are removed and verified helplines are attached instead
- The ML model can raise the safety level but never lower it, and the rules work without it

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
| `backend/safety.py` | Rule gate (phrases, leetspeak, misspellings, denials, context), `decide()`, crisis text, verified helplines |
| `backend/ml.py` | Multi-task model (shared encoder + 3 heads), shared by training and inference |
| `train.py` | Fine-tuning, model selection, test metrics |
| `backend/memory.py` | ChromaDB store of de-identified session notes |
| `backend/llm.py` | Gemini reply prompt and modes, de-identified summaries, model fallback |
| `backend/app.py` | FastAPI pipeline + SQLite sessions; serves `frontend/dist` |
| `frontend/src/App.jsx` | The whole UI |
| `tests/` | Safety rule tests and pipeline / API tests |

Privacy: when a session ends, its transcript is deleted. Only a de-identified note (for memory) and mood stats remain, locally.
HIPAA-inspired data minimization principles, not HIPAA compliance.
