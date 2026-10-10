"""Multi-task Transformer: shared encoder -> emotion / intensity / risk heads.
Used by train.py (training) and app.py (inference). If no trained model exists,
load() returns None and the app runs on the rule-based safety gate alone."""
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "lumina"
RISK_LABELS = ["safe", "caution", "critical"]


class LuminaModel(nn.Module):
    def __init__(self, encoder_name, n_emotions, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.emotion = nn.Linear(hidden, n_emotions)
        self.intensity = nn.Linear(hidden, 1)
        self.risk = nn.Linear(hidden, len(RISK_LABELS))

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        h = self.dropout(h)
        return self.emotion(h), torch.sigmoid(self.intensity(h)).squeeze(-1), self.risk(h)


def save(model, tokenizer, emotions, out_dir, extra=None):
    out = Path(out_dir)
    (out / "encoder").mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out / "encoder")
    tokenizer.save_pretrained(out / "encoder")
    heads = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
    torch.save(heads, out / "heads.pt")
    (out / "config.json").write_text(json.dumps({"emotions": emotions, "risk": RISK_LABELS, **(extra or {})}, indent=2))


def load_heads(model, model_dir):
    """Encoder weights come from from_pretrained; this loads the three heads and checks they all fit."""
    res = model.load_state_dict(torch.load(Path(model_dir) / "heads.pt", map_location="cpu"), strict=False)
    missing = [k for k in res.missing_keys if not k.startswith("encoder.")]
    if missing or res.unexpected_keys:
        raise ValueError(f"heads.pt does not match the model: missing={missing} unexpected={res.unexpected_keys}")


class Analyzer:
    def __init__(self, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "config.json").read_text())
        self.emotions = self.config["emotions"]
        self.max_len = self.config.get("max_len", 128)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir / "encoder")
        self.model = LuminaModel(model_dir / "encoder", len(self.emotions))
        load_heads(self.model, model_dir)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text):
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len, return_tensors="pt")
        emo, intensity, risk = self.model(enc["input_ids"], enc["attention_mask"])
        emo, risk = emo.softmax(-1)[0], risk.softmax(-1)[0]
        return {
            "emotion": self.emotions[int(emo.argmax())],
            "emotion_confidence": round(float(emo.max()), 3),
            "intensity": round(float(intensity[0]), 3),
            "risk": RISK_LABELS[int(risk.argmax())],
            "risk_confidence": round(float(risk.max()), 3),
        }


def load(model_dir=MODEL_DIR):
    if not (Path(model_dir) / "heads.pt").exists():
        return None
    return Analyzer(model_dir)
