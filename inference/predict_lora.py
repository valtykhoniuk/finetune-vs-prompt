import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baselines"))

from evaluate import compute_metrics
from io_utils import load_jsonl
from labels import LABELS

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_DIR = ROOT / "models" / "foxschool-intent-lora"
TEST_PATH = ROOT / "data/test.jsonl"
RESULTS_PATH = ROOT / "results/lora.json"

INSTRUCTION = (
    "Classify this FoxSchool support ticket. "
    "Reply with only one label: billing, bug, how-to, refund, other."
)

PROMPT = """### Instruction:
{}

### Input:
{}

### Response:
"""

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def resolve_adapter_dir() -> Path:
    candidates = [
        ADAPTER_DIR,
        ROOT / "models",
    ]
    for path in candidates:
        if (path / "adapter_config.json").exists():
            return path
    raise FileNotFoundError(
        f"adapter_config.json not found. Expected: {ADAPTER_DIR}\n"
        "Unzip from Colab: unzip ~/Downloads/foxschool-intent-lora.zip -d models/"
    )


def load_model():
    device = get_device()
    dtype = torch.float16 if device != "cpu" else torch.float32
    adapter_dir = resolve_adapter_dir()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map=device,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model, tokenizer, device

def predict_label(model, tokenizer, device, text: str) -> str:
    prompt = PROMPT.format(INSTRUCTION, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated.split("### Response:")[-1].strip().lower()

    for label in LABELS:
        if label in answer:
            return label
    return "other"

def main():
    model, tokenizer, device = load_model()
    test_rows = load_jsonl(TEST_PATH)
    predictions = []

    for i, row in enumerate(test_rows, 1):
        pred = predict_label(model, tokenizer, device, row["text"])
        predictions.append({
            "text": row["text"],
            "gold": row["label"],
            "pred": pred,
        })
        print(f"[{i}/{len(test_rows)}] gold={row['label']} pred={pred}")

    metrics = compute_metrics(predictions)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "predictions": predictions}, f, indent=2)

    print(f"\nLoRA accuracy: {metrics['accuracy']}")
    print(f"Saved → {RESULTS_PATH}")

if __name__ == "__main__":
    main()