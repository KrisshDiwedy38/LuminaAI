# CLAUDE.md

## Project state

Local demo build. Design source: `lumina-project-plan.md` (mentor proposal). Deviations for the demo: SQLite instead of Supabase (no auth), plain Python pipeline instead of CrewAI, ChromaDB's local MiniLM embeddings instead of Gemini embeddings.

Commands (Windows, venv at `.venv`): `python -m pytest` · `uvicorn backend.app:app` (serves `frontend/dist` at :8000) · `cd frontend; npm run build` / `npm run dev` · `python train.py --data <csv>` → `models/lumina/`.

## What Lumina is

Gen-Z mental wellness chat companion (portfolio/mentor project). Per message:

1. **Deterministic safety gate** (keyword/phrase rules + misspelling/leetspeak handling) → SAFE / CAUTION / CRITICAL
2. **Fine-tuned multi-task Transformer** (DistilBERT/RoBERTa/etc.) → emotion label, intensity (0–1 regression), risk (SAFE/CAUTION/CRITICAL)
3. **Safety decision** combines 1 + 2. CRITICAL → crisis response, **skip the rest of the pipeline**.
4. **Memory**: retrieve relevant past de-identified session summaries (vector search)
5. **Therapist agent** (Gemini) writes the reply from message + ML signals + memory + history (+ optional Gen-Z tone)

Philosophy: ML for perception, deterministic code for safety, retrieval for memory, LLM only for language.

Planned stack: Python/FastAPI, PyTorch + HF Transformers, CrewAI, Gemini, ChromaDB, Supabase (Postgres/auth), React + Tailwind, Render (backend), Vercel (frontend).

## Non-negotiables

- The rule-based safety gate must work **independently of the ML model** (model down/uncertain ≠ unsafe pass-through). ML risk is an input to the safety decision, never the only one.
- CRITICAL must short-circuit: no memory lookup, no Gemini call, no Gen-Z tone on crisis wording. Crisis text is fixed, not generated.
- The safety gate reads context, not just the current message: vague wishes ("i wanna feel that") after recent talk of death are CRITICAL.
- Helpline numbers and sites come only from `CRISIS_RESOURCES` (India only). Gemini output is filtered by `strip_unverified_numbers`; it once invented 988 (US).
- Safety false negatives (actual CRITICAL → predicted SAFE) are the metric that matters most; track critical recall separately from accuracy.
- Every safety rule change needs a test case in `tests/test_safety_rules.py`.
- Never claim diagnosis, treatment, therapist replacement, or HIPAA compliance. Wording: "HIPAA-inspired data minimization principles". Keep the visible "not a replacement for professional care / emergency services" disclaimer.
- Do not store PII beyond what's needed; memory stores de-identified summaries, not raw transcripts.
- Test set stays unseen during training and hyperparameter tuning.
- Example metrics in the plan are illustrative — never present them as results.

## Planned layout

Tests go in `tests/` (see plan §18 for intended file names). Use pytest.
