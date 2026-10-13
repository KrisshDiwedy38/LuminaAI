"""Fine-tune Lumina's multi-task model (emotion + intensity + risk).

    python train.py --data data/lumina.csv

CSV columns: text, emotion, intensity (0-1), risk (safe|caution|critical), optional split (train|val|test).
With no split column the data is split 70/15/15, stratified by risk. The best epoch is chosen on
validation only; the test set is scored once, at the end, on the saved model.
Writes models/lumina/ (picked up by the app on restart) and models/lumina/metrics.json.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from backend.ml import MODEL_DIR, RISK_LABELS, LuminaModel, load_heads, save

CRITICAL = RISK_LABELS.index("critical")
SAFE = RISK_LABELS.index("safe")


def load_data(path):
    df = pd.read_csv(path)
    missing = {"text", "emotion", "intensity", "risk"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["text", "emotion", "intensity", "risk"]).copy()
    df["text"] = df["text"].astype(str)
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    df["risk"] = df["risk"].astype(str).str.strip().str.lower()
    df["intensity"] = pd.to_numeric(df["intensity"], errors="raise")
    bad = ~df["risk"].isin(RISK_LABELS)
    if bad.any():
        raise SystemExit(f"{bad.sum()} rows have risk outside {RISK_LABELS}, e.g. {df['risk'][bad].iloc[0]!r}")
    if not df["intensity"].between(0, 1).all():
        raise SystemExit("intensity must be between 0 and 1")
    return df


def split(df, seed):
    if "split" in df.columns:
        s = df["split"].astype(str).str.strip().str.lower()
        parts = df[s == "train"], df[s.isin(["val", "valid", "validation", "dev"])], df[s == "test"]
    else:
        strat = df["risk"] if df["risk"].value_counts().min() >= 3 else None
        train, rest = train_test_split(df, test_size=0.3, stratify=strat, random_state=seed)
        strat = rest["risk"] if strat is not None and rest["risk"].value_counts().min() >= 2 else None
        parts = (train, *train_test_split(rest, test_size=0.5, stratify=strat, random_state=seed))
    if any(len(p) == 0 for p in parts):
        raise SystemExit(f"train/val/test sizes {[len(p) for p in parts]}: every split needs rows")
    return parts


def is_better(best, cr, score, max_cr, tol):
    """Model selection. Critical recall first, but within `tol` (one validation crisis example) of the best
    recall seen so far counts as a tie and the overall score decides. Comparing with the best ever seen
    stops small drops adding up across epochs. best: (critical_recall, score) of the saved epoch, or None."""
    return best is None or best[0] < max_cr - tol or (cr >= max_cr - tol and score > best[1])


def batches(df, tok, emo_idx, size, max_len, shuffle=False):
    order = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
    for i in range(0, len(df), size):
        b = df.iloc[order[i:i + size]]
        enc = tok(list(b["text"]), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        yield (enc["input_ids"], enc["attention_mask"],
               torch.tensor([emo_idx[e] for e in b["emotion"]]),
               torch.tensor(b["intensity"].values, dtype=torch.float),
               torch.tensor([RISK_LABELS.index(r) for r in b["risk"]]))


@torch.no_grad()
def evaluate(model, df, tok, emotions, args, dev, report=False):
    model.eval()
    emo_idx = {e: i for i, e in enumerate(emotions)}
    E, I, R = [], [], []
    for ids, mask, *_ in batches(df, tok, emo_idx, args.batch_size, args.max_len):
        e, i, r = model(ids.to(dev), mask.to(dev))
        E += e.argmax(-1).tolist()
        I += i.tolist()
        R += r.argmax(-1).tolist()
    ye = [emo_idx[e] for e in df["emotion"]]
    yr = [RISK_LABELS.index(r) for r in df["risk"]]
    yi = df["intensity"].to_numpy()
    actual_crit = [k for k, y in enumerate(yr) if y == CRITICAL]
    pred_crit = [k for k, p in enumerate(R) if p == CRITICAL]
    err = np.array(I) - yi
    metrics = {
        "n": len(df),
        "emotion": {
            "accuracy": float(np.mean(np.array(E) == ye)),
            "macro_f1": f1_score(ye, E, average="macro", labels=range(len(emotions)), zero_division=0),
            "weighted_f1": f1_score(ye, E, average="weighted", labels=range(len(emotions)), zero_division=0),
            "labels": emotions,
            "confusion_matrix": confusion_matrix(ye, E, labels=range(len(emotions))).tolist(),
        },
        "risk": {
            "accuracy": float(np.mean(np.array(R) == yr)),
            "macro_f1": f1_score(yr, R, average="macro", labels=range(3), zero_division=0),
            "critical_recall": sum(R[k] == CRITICAL for k in actual_crit) / len(actual_crit) if actual_crit else None,
            "critical_precision": sum(yr[k] == CRITICAL for k in pred_crit) / len(pred_crit) if pred_crit else None,
            # the failure that matters most: an actual crisis the model calls safe
            "critical_predicted_safe": sum(R[k] == SAFE for k in actual_crit),
            "labels": RISK_LABELS,
            "confusion_matrix": confusion_matrix(yr, R, labels=range(3)).tolist(),
        },
        "intensity": {
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "pearson": float(np.corrcoef(I, yi)[0, 1]) if len(yi) > 1 and np.std(yi) > 0 and np.std(I) > 0 else None,
        },
    }
    if report:
        print(classification_report(ye, E, labels=range(len(emotions)), target_names=emotions, zero_division=0))
        print(classification_report(yr, R, labels=range(3), target_names=RISK_LABELS, zero_division=0))
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="distilbert-base-uncased", help="any Hugging Face encoder, e.g. roberta-base")
    ap.add_argument("--out", default=str(MODEL_DIR))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lambda-intensity", type=float, default=1.0)
    ap.add_argument("--lambda-risk", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_data(args.data)
    train_df, val_df, test_df = split(df, args.seed)
    emotions = sorted(df["emotion"].unique())
    emo_idx = {e: i for i, e in enumerate(emotions)}
    print(f"device={dev} train={len(train_df)} val={len(val_df)} test={len(test_df)} emotions={emotions}")
    print("risk counts (train):", train_df["risk"].value_counts().to_dict())

    tok = AutoTokenizer.from_pretrained(args.model)
    model = LuminaModel(args.model, len(emotions), args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * math.ceil(len(train_df) / args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    # Inverse-frequency risk weights: rare CRITICAL rows weigh more, which favours critical recall.
    counts = train_df["risk"].value_counts()
    risk_w = torch.tensor([len(train_df) / (3 * counts.get(r, 1)) for r in RISK_LABELS], dtype=torch.float, device=dev)
    ce_emotion, ce_risk, mse = nn.CrossEntropyLoss(), nn.CrossEntropyLoss(weight=risk_w), nn.MSELoss()

    n_crit = int((val_df["risk"] == "critical").sum())
    tol = (1 / n_crit if n_crit else 0.0) + 1e-9  # float slack: exactly one example must still tie
    best, max_cr = None, 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for ids, mask, ye, yi, yr in batches(train_df, tok, emo_idx, args.batch_size, args.max_len, shuffle=True):
            e, i, r = model(ids.to(dev), mask.to(dev))
            loss = (ce_emotion(e, ye.to(dev))
                    + args.lambda_intensity * mse(i, yi.to(dev))
                    + args.lambda_risk * ce_risk(r, yr.to(dev)))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
        val = evaluate(model, val_df, tok, emotions, args, dev)
        cr = val["risk"]["critical_recall"]
        print(f"epoch {epoch}: loss={total / n:.4f} | val emotion_f1={val['emotion']['macro_f1']:.3f} "
              f"risk_f1={val['risk']['macro_f1']:.3f} critical_recall={'n/a' if cr is None else f'{cr:.3f}'} "
              f"intensity_mae={val['intensity']['mae']:.3f}")
        # Safety first: critical recall (one example of slack), then emotion + risk F1 - intensity error.
        score = val["emotion"]["macro_f1"] + val["risk"]["macro_f1"] - val["intensity"]["mae"]
        max_cr = max(max_cr, cr or 0.0)
        if is_better(best, cr or 0.0, score, max_cr, tol):
            best = (cr or 0.0, score)
            save(model, tok, emotions, args.out,
                 {"base_model": args.model, "max_len": args.max_len, "epoch": epoch, "val": val})
            print(f"  saved best model to {args.out}")

    # The test set is touched exactly once, on the model chosen by validation.
    best = LuminaModel(Path(args.out) / "encoder", len(emotions)).to(dev)
    load_heads(best, args.out)
    print("\nTEST SET")
    test = evaluate(best, test_df, tok, emotions, args, dev, report=True)
    cfg = json.loads((Path(args.out) / "config.json").read_text())
    metrics = {"base_model": args.model, "best_epoch": cfg["epoch"], "args": vars(args), "val": cfg["val"], "test": test}
    (Path(args.out) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"critical recall={test['risk']['critical_recall']} | critical predicted SAFE={test['risk']['critical_predicted_safe']}"
          f" | intensity MAE={test['intensity']['mae']:.3f}\nmetrics written to {Path(args.out) / 'metrics.json'}"
          "\nRestart the backend to load the new model.")


if __name__ == "__main__":
    main()
