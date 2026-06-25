import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from labels import LABELS, LABEL_DESCRIPTIONS
from io_utils import load_jsonl
from evaluate import compute_metrics

ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = ROOT / "data/test.jsonl"
RESULTS_PATH = ROOT / "results/zero_shot.json"

load_dotenv(ROOT / ".env")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def build_system_prompt() -> str:
    lines = [
        "You classify FoxSchool support tickets into exactly one label.",
        "Reply with ONLY the label word, nothing else.",
        "Valid labels:",
    ]
    for label in LABELS:
        lines.append(f"- {label}: {LABEL_DESCRIPTIONS[label]}")
    return "\n".join(lines)

def predict_label(text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": text}
        ]
    )
    raw = resp.choices[0].message.content.strip().lower()

    for label in LABELS:
        if label in raw:
            return label
    return 'other'

def main():
    test_rows = load_jsonl(TEST_PATH)
    predictions = []

    for i, row in enumerate(test_rows, 1):
        pred = predict_label(row["text"])
        predictions.append({
            "text": row["text"],
            "gold": row["label"],
            "pred": pred
        })
        print(f"[{i}/{len(test_rows)}] gold={row['label']} pred={pred}")
        time.sleep(0.2)
    metrics = compute_metrics(predictions)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "predictions": predictions}, f, indent=2)
    print(f"\nZero-shot accuracy: {metrics['accuracy']}")
    print(f"Saved → {RESULTS_PATH}")

if __name__ == "__main__":
    main()