import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_IN = ROOT / "data/train.jsonl"
TRAIN_OUT = ROOT / "training/train_alpaca.jsonl"

INSTRUCTION = (
    "Classify this FoxSchool support ticket. "
    "Reply with only one label: billing, bug, how-to, refund, other."
)

def main():
    rows_out = []
    with TRAIN_IN.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_out.append({
                "instruction": INSTRUCTION,
                "input": row["text"],
                "output": row["label"]
            })
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_OUT.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows_out)} rows → {TRAIN_OUT}")

if __name__ == "__main__":
    main()