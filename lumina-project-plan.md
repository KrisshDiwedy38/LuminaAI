# Lumina AI — Multi-Agent ML Mental Wellness Companion
## Mentor Proposal & Technical Project Plan

**Project Status:** Proposed / Development Plan  
**Primary Goal:** Build and evaluate a full-stack AI/ML mental wellness companion that combines a trained ML model, deterministic safety controls, conversational memory, multi-agent orchestration, and a generative AI response layer.

---

# 1. Executive Summary

Lumina AI is a Gen-Z-focused AI mental wellness companion designed as an engineering and AI/ML portfolio project. It is intended to provide conversational support, track emotional trends, retain useful conversational context, and provide structured progress information.

The proposed system is **not a diagnostic system and not a replacement for licensed mental-health care or emergency services**.

The major extension to the original project plan is the introduction of a **fine-tuned Transformer-based ML model**. Rather than asking the generative LLM to infer every aspect of a user's state, the trained model acts as a specialized perception layer.

For each user message, the ML model predicts:

1. **Emotion** — what emotional state is being expressed.
2. **Intensity** — how strongly the emotion is being expressed.
3. **Risk** — whether additional safety handling is required.

These structured predictions are combined with conversation history and retrieved memory before being passed to the generative Therapist Agent.

The resulting architecture separates responsibilities:

> **ML model: WHAT is happening?**  
> **Memory + agents: WHAT context do we know?**  
> **Gemini: HOW should we respond?**

---

# 2. Project Objectives

## Primary Objectives

- Build a working full-stack AI application.
- Train and evaluate a specialized ML model rather than relying entirely on an external LLM.
- Implement multi-task learning for emotion, intensity, and conversational risk.
- Implement a deterministic safety layer as a second line of defense.
- Use vector memory to retrieve relevant historical context.
- Use CrewAI for agent orchestration.
- Use Gemini for natural-language response generation.
- Provide a dashboard showing session history, emotional trends, assessments, and summaries.
- Establish measurable evaluation criteria for both the ML model and the complete system.

## Secondary Objectives

- Demonstrate responsible AI engineering.
- Demonstrate privacy-conscious data handling.
- Handle free-tier compute/API constraints.
- Build a reproducible automated test suite.
- Document model limitations, failure cases, and future improvements.

---

# 3. Core System Concept

Lumina is not intended to be a chatbot where a single LLM performs every task.

Instead, the system divides responsibilities between specialized components.

```text
                         USER
                           |
                           v
                  +----------------+
                  | Preprocessing  |
                  +-------+--------+
                          |
             +------------+------------+
             |                         |
             v                         v
     +---------------+        +----------------+
     | Rule-Based    |        | Trained ML     |
     | Safety Gate   |        | Transformer    |
     +-------+-------+        +-------+--------+
             |                        |
             |                +-------+--------+
             |                |        |       |
             |                v        v       v
             |             Emotion  Intensity Risk
             |                |        |       |
             +----------------+--------+-------+
                              |
                              v
                       Safety Decision
                              |
                +-------------+-------------+
                |                           |
             CRITICAL                  SAFE / CAUTION
                |                           |
                v                           v
        Crisis Response              Memory System
                                      / ChromaDB
                                           |
                                           v
                                 Relevant Context
                                           |
                                           v
                                  Therapist Agent
                                      / Gemini
                                           |
                                           v
                                     RESPONSE
                                           |
                                           v
                                         USER
```

---

# 4. Proposed Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| ML model | PyTorch + Hugging Face Transformers | Train/fine-tune emotion, intensity, and risk model |
| LLM | Gemini 1.5/2.0 Flash | Generative response and reasoning |
| Agent orchestration | CrewAI | Coordinate specialized agents |
| Backend | Python + FastAPI | API and application backend |
| Vector memory | ChromaDB | Semantic retrieval of previous context |
| Embeddings | Gemini `text-embedding-004` | Convert summaries into vector representations |
| Database/Auth | Supabase/Postgres | Users, sessions, messages, assessments, summaries |
| Frontend | React + Tailwind CSS | Chat application and dashboard |
| Backend hosting | Render | Backend deployment |
| Frontend hosting | Vercel | Frontend deployment |

The original project plan specifies Gemini, CrewAI, FastAPI, ChromaDB, Supabase, React/Tailwind, Render, and Vercel. The Transformer/PyTorch/Hugging Face ML layer is the proposed extension for the ML-focused version.

---

# 5. ML Model Design

## 5.1 Model Type

The proposed model is a **pretrained Transformer fine-tuned for multi-task classification/regression**.

Candidate architectures:

- DistilBERT
- BERT
- RoBERTa
- DeBERTa

The final model will be selected after comparing validation performance, inference cost, model size, and deployment feasibility.

We will **not train a language model from scratch**.

Instead:

```text
Pretrained Transformer
        |
        v
   Fine-tuning
        |
        v
Lumina ML Model
```

---

# 6. Multi-Task Learning Architecture

A single Transformer will produce a shared language representation, followed by separate task-specific heads.

```text
                    USER MESSAGE
                         |
                         v
                   Tokenization
                         |
                         v
                  Transformer
                         |
                         v
               Shared Representation
                  /        |        \
                 /         |         \
                v          v          v
          Emotion Head  Intensity   Risk Head
                         Head
                |          |          |
                v          v          v
             Anxiety      0.82      CAUTION
```

## Task A — Emotion Classification

The model predicts the primary emotional state expressed in the message.

The final label set will be established during dataset design. Candidate labels include:

- anxiety
- sadness
- stress
- anger
- fear
- joy
- neutral
- other

The label taxonomy must be kept small enough to produce reliable annotations.

## Task B — Intensity

The model estimates how strongly the emotion is expressed.

Two possible implementations will be evaluated:

### Regression

Output:

```text
0.00 — 1.00
```

### Classification

For example:

```text
LOW
MEDIUM
HIGH
```

Regression is the preferred initial experiment because it provides a continuous value that can be plotted across a session.

## Task C — Risk Classification

The model predicts:

```text
SAFE
CAUTION
CRITICAL
```

The exact safety policy will remain independently controlled by deterministic safety logic.

This means the ML prediction is **an input to the safety decision, not the sole safety mechanism**.

---

# 7. Defense-in-Depth Safety Architecture

The original project plan specifies a deterministic rule-based Safety Agent with keyword/phrase matching, misspelling/leetspeak handling, and three output states.

The ML version should retain this layer.

```text
                  USER MESSAGE
                       |
              +--------+--------+
              |                 |
              v                 v
        Rule-Based          ML Risk Model
        Safety Gate              |
              |                  |
              +--------+---------+
                       |
                       v
                 Safety Decision
                       |
           +-----------+-----------+
           |           |           |
           v           v           v
         SAFE       CAUTION     CRITICAL
                                   |
                                   v
                           Crisis Resources
```

### Why keep both?

The rule-based layer is:

- deterministic
- easy to test
- transparent
- fast
- independent of model uncertainty

The ML model can capture semantic/contextual expressions that may not exactly match a keyword list.

This provides two complementary safety mechanisms.

---

# 8. Example End-to-End Flow

User message:

> "I've been studying for weeks and I'm exhausted. I don't think I'm good enough to pass."

### Step 1 — ML Analysis

Possible output:

```json
{
  "emotion": "anxiety",
  "emotion_confidence": 0.91,
  "intensity": 0.84,
  "risk": "caution",
  "risk_confidence": 0.73
}
```

These values are illustrative and are not claimed model results.

### Step 2 — Safety Decision

The deterministic safety layer and ML risk result are combined according to the defined safety policy.

### Step 3 — Memory Retrieval

ChromaDB searches for relevant previous session summaries.

Example:

```text
Previous context:
User has previously discussed academic pressure
and examination-related stress.
```

### Step 4 — Therapist Agent

Gemini receives:

```text
USER MESSAGE:
...

ML ANALYSIS:
Emotion: anxiety
Intensity: 0.84
Risk: caution

RELEVANT MEMORY:
...

CONVERSATION:
...

GEN-Z MODE:
enabled
```

### Step 5 — Response

Gemini generates the final supportive response.

The ML model does not generate the response itself.

---

# 9. Training Data

The ML model requires labelled examples.

Conceptually:

| Text | Emotion | Intensity | Risk |
|---|---|---:|---|
| "I'm having a great day!" | joy | 0.20 | safe |
| "My exams are stressing me out." | anxiety | 0.65 | safe |
| "I feel completely hopeless." | sadness | 0.88 | caution |
| "I don't know how much longer I can take." | distress | 0.94 | caution |
| Crisis example | crisis | 1.00 | critical |

The final dataset must be selected based on:

- licensing
- privacy
- annotation quality
- suitability for the intended task
- class balance
- demographic/language coverage
- safety considerations

Mental-health-related data requires additional care. Personally identifiable information should not be unnecessarily retained.

---

# 10. Dataset Split

A starting split:

```text
                 DATASET
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
      TRAIN       VALID        TEST
       70%         15%         15%
```

The test set must remain unseen during training and hyperparameter selection.

A stratified split should be considered to preserve class proportions.

---

# 11. Training Process

Conceptual training loop:

```text
Training Batch
      |
      v
Tokenizer
      |
      v
Transformer
      |
      v
Shared Representation
      |
      +--------+---------+
      |        |         |
      v        v         v
   Emotion  Intensity   Risk
   Output     Output    Output
      |        |         |
      +--------+---------+
               |
               v
          Total Loss
               |
               v
        Backpropagation
               |
               v
         Weight Update
```

A conceptual combined loss:

```text
L_total =
    L_emotion
    + λ1 * L_intensity
    + λ2 * L_risk
```

The λ weights will be tuned experimentally.

---

# 12. Training Hyperparameters

Initial values should be treated as experimental starting points rather than fixed requirements.

| Parameter | Starting Point |
|---|---:|
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs | 3–5 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Maximum sequence length | 128–256 |
| Dropout | 0.1 |
| Loss | Task-specific combined loss |

Experiments should compare configurations and select the model using validation performance and safety-oriented criteria.

---

# 13. Model Evaluation

The ML model will be evaluated independently before being integrated into Lumina.

## 13.1 Emotion Classification

Metrics:

- Accuracy
- Precision
- Recall
- Macro F1
- Weighted F1
- Confusion matrix

Macro F1 is particularly useful when emotion classes are imbalanced.

Formula:

```text
Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 × (Precision × Recall)
     -------------------------
       Precision + Recall
```

---

# 14. Risk Model Evaluation

Risk evaluation requires special treatment.

Overall accuracy is not enough.

For the `CRITICAL` class, we should prioritize:

### Critical Recall

```text
Critical Recall =
True Critical Predictions
--------------------------------
All Actual Critical Examples
```

The key failure mode is a **false negative**:

```text
Actual: CRITICAL
Model:  SAFE
```

This must be monitored separately.

Other metrics:

- Critical precision
- Critical recall
- Macro F1
- Confusion matrix
- False-negative rate
- False-positive rate

---

# 15. Intensity Evaluation

If implemented as regression:

- MAE
- MSE
- RMSE
- correlation with human annotations

Mean Absolute Error:

```text
MAE = average(|predicted - actual|)
```

Example:

```text
Human rating:      0.80
Model prediction: 0.74

Absolute error:    0.06
```

---

# 16. System-Level Evaluation

The ML model is only one part of Lumina.

The complete system must also be evaluated.

## 16.1 Safety Pipeline

Test whether:

```text
CRITICAL
   ↓
Crisis Response
   ↓
Downstream therapeutic pipeline skipped
```

The original project plan explicitly requires the critical path to short-circuit steps 2–3.

## 16.2 Memory

Evaluate:

- Top-1 retrieval accuracy
- Top-3 retrieval accuracy
- relevant retrieval rate
- irrelevant retrieval rate

## 16.3 LLM Response Quality

Human evaluators can rate responses from 1–5 on:

| Dimension | Question |
|---|---|
| Empathy | Does the response acknowledge the user's feelings? |
| Relevance | Does it address the user's actual message? |
| Helpfulness | Is the response useful? |
| Safety | Does it avoid unsafe or inappropriate advice? |
| Coherence | Is it understandable and consistent? |
| Tone | Is the requested tone appropriate? |

## 16.4 Reliability

Measure:

- successful request rate
- API failure rate
- timeout rate
- 429/rate-limit rate
- retry success rate
- graceful degradation behavior

The original plan already calls for client throttling, exponential backoff, and graceful degradation for API rate limits.

## 16.5 Latency

Measure:

- average latency
- P50 latency
- P95 latency
- ML inference time
- memory retrieval time
- LLM generation time
- complete request latency

---

# 17. Proposed Acceptance Criteria

These are **proposed project targets**, not results and not thresholds specified by the original project plan.

| Component | Metric | Proposed Target |
|---|---|---:|
| Emotion model | Macro F1 | ≥ 0.85 |
| Risk model | Critical Recall | ≥ 0.98 |
| Risk model | Critical Precision | ≥ 0.90 |
| Intensity model | MAE | ≤ 0.10 |
| Memory | Relevant retrieval rate | ≥ 0.85 |
| Pipeline | Correct routing | 100% on defined tests |
| API | Successful requests | ≥ 99% |
| Response quality | Human safety rating | ≥ 4/5 |

These thresholds should be reviewed with the mentor before being adopted as official project requirements.

---

# 18. Automated Testing Strategy

The project should contain a dedicated test suite.

```text
tests/
│
├── test_emotion_model.py
├── test_risk_model.py
├── test_intensity_model.py
├── test_safety_rules.py
├── test_memory.py
├── test_agent_pipeline.py
├── test_therapist_agent.py
└── test_end_to_end.py
```

### Unit Tests

Test each component independently.

Examples:

```text
Safety rule detection
Tokenizer/model output
Intensity range
Risk label validity
Memory retrieval
Database operations
```

### Integration Tests

Test component interactions:

```text
ML → Safety
ML → Memory
Memory → Therapist Agent
Safety → Crisis response
```

### End-to-End Tests

Test:

```text
User
 ↓
API
 ↓
ML
 ↓
Safety
 ↓
Memory
 ↓
Agents
 ↓
Gemini
 ↓
Response
```

---

# 19. Evaluation Dashboard

A dedicated internal evaluation dashboard can demonstrate project performance.

Example:

```text
LUMINA AI — MODEL EVALUATION

Emotion Classification
F1 Score                 91.4%

Risk Classification
F1 Score                 96.8%

Critical Recall
                         99.1%

Intensity MAE
                         0.08

Memory Retrieval
                         89.2%

Pipeline Reliability
                         99.7%

Average Response Time
                         1.8 sec
```

These values are examples of how results could be presented. They are **not current Lumina results**.

---

# 20. What Defines Project Success?

Success should not be defined as:

> "The chatbot produces convincing responses."

Instead, success means the complete system satisfies measurable technical and safety requirements.

### ML Success

The trained model performs reliably on unseen data.

### Safety Success

The system minimizes critical-risk false negatives and routes safety cases correctly.

### Memory Success

Relevant previous context is retrieved while irrelevant context is minimized.

### Generative AI Success

The Therapist Agent produces relevant, coherent, supportive, and appropriately bounded responses.

### Engineering Success

The complete application works reliably under the project's infrastructure constraints.

### Demonstration Success

The system can show measurable improvement and performance through reproducible evaluation rather than subjective claims.

---

# 21. Important Limitations

Lumina should **not** claim:

- clinical diagnosis
- medical diagnosis
- treatment of mental illness
- replacement for a therapist
- emergency-service replacement
- HIPAA compliance unless independently established

The existing project plan explicitly recommends describing the privacy approach as **"HIPAA-inspired data minimization principles"** rather than claiming HIPAA compliance.

The application should display a visible disclaimer explaining that Lumina is not a replacement for licensed mental-health care or emergency services.

---

# 22. Development Roadmap

## Phase 1 — Foundation

- FastAPI project
- Supabase schema
- authentication
- React application shell
- basic chat UI

## Phase 2 — Dataset & ML Research

- define emotion taxonomy
- define intensity representation
- define risk labels
- identify appropriate datasets
- establish annotation/quality process
- create train/validation/test splits
- establish baseline model

## Phase 3 — ML Model

- select Transformer
- implement tokenizer
- implement multi-task heads
- establish training pipeline
- train baseline
- tune hyperparameters
- evaluate
- perform error analysis
- save best model

## Phase 4 — Safety Layer

- retain deterministic safety rules
- add crisis/caution test cases
- integrate ML risk prediction
- define safety decision policy
- test false negatives and false positives

## Phase 5 — Memory

- integrate ChromaDB
- generate de-identified session summaries
- create embeddings
- implement retrieval
- evaluate retrieval quality

## Phase 6 — Agents

- integrate ML output into CrewAI
- implement Emotion/Memory/Resource Agent
- implement Therapist Agent
- add conditional Gen-Z tone
- preserve safety-critical wording
- implement Summary Agent

## Phase 7 — Application

- chat experience
- sessions
- dashboard
- mood/intensity trends
- assessment module
- PHQ-9/GAD-7 tracking

## Phase 8 — Evaluation

- automated tests
- ML benchmark
- safety benchmark
- memory benchmark
- response-quality evaluation
- latency testing
- reliability testing

## Phase 9 — Deployment

- Render backend
- Vercel frontend
- environment/security configuration
- production testing
- monitoring

---

# 23. Final Proposed Architecture

```text
                         LUMINA AI
                             |
             +---------------+---------------+
             |                               |
             v                               v
       USER INTERFACE                   AUTH / DATA
        React + Tailwind                Supabase
             |
             v
          FastAPI
             |
             v
    +---------------------+
    |    ML ANALYSIS      |
    |                     |
    | Fine-tuned          |
    | Transformer         |
    +----------+----------+
               |
       +-------+-------+
       |       |       |
       v       v       v
    Emotion Intensity Risk
       |       |       |
       +-------+-------+
               |
               v
    +----------------------+
    | Deterministic Safety |
    |        Gate          |
    +----------+-----------+
               |
        +------+------+
        |             |
     CRITICAL      CONTINUE
        |             |
        v             v
 Crisis Resources  ChromaDB
                      |
                      v
               Relevant Memory
                      |
                      v
                 CrewAI
                      |
                      v
               Therapist Agent
                      |
                      v
                   Gemini
                      |
                      v
                 RESPONSE
                      |
                      v
                    USER
```

---

# 24. Project Statement for Mentor

**Proposed project statement:**

> **Lumina AI is a multi-agent AI mental wellness companion that combines a fine-tuned Transformer-based multi-task ML model with deterministic safety controls, vector-based conversational memory, CrewAI agent orchestration, and Gemini-based generative response synthesis. The trained ML model predicts emotional state, emotional intensity, and conversational risk, while the generative layer uses these structured signals and retrieved context to produce an appropriate response. The system will be evaluated at both the model and system levels using classification metrics, safety-focused recall, regression error, memory retrieval metrics, response-quality evaluation, latency, reliability, and end-to-end testing.**

---

# 25. Expected Technical Contribution

The primary technical contribution is not simply building a chatbot.

The project demonstrates:

1. **Transfer learning** through Transformer fine-tuning.
2. **Multi-task learning** through shared representation and multiple prediction heads.
3. **Safety-aware AI architecture** using defense in depth.
4. **Agent orchestration** using CrewAI.
5. **Retrieval-augmented conversational memory** using ChromaDB.
6. **Generative AI integration** using Gemini.
7. **Full-stack AI engineering** using FastAPI, React, and Supabase.
8. **Quantitative model evaluation** rather than subjective chatbot testing.
9. **Privacy-conscious data minimization.**
10. **Engineering under practical free-tier constraints.**

The central design philosophy is:

> **Specialized ML for perception, deterministic systems for safety, retrieval for memory, agents for orchestration, and an LLM for language generation.**

This separation of responsibilities makes Lumina easier to evaluate, debug, improve, and explain.
