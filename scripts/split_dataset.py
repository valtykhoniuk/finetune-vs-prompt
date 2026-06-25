import json
import random
from pathlib import Path
from collections import Counter

INPUT = Path("data/tickets_labeled.jsonl")
TRAIN = Path("data/train.jsonl")
TEST = Path("data/test.jsonl")
TEST_RATIO = 0.2
SEED = 42

def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def main():
    rows = load_rows(INPUT)
    random.seed(SEED)
    random.shuffle(rows)

    split_at = int(len(rows) * (1-TEST_RATIO))
    train, test = rows[:split_at], rows[split_at:]
    for path, data in [(TRAIN, train), (TEST, test)]:
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Total: {len(rows)}")
    print(f"Train: {len(train)} → {TRAIN}")
    print(f"Test:  {len(test)} → {TEST}")
    print("Train labels:", Counter(r["label"] for r in train))
    print("Test labels:", Counter(r["label"] for r in test))

if __name__ == "__main__":
    main()