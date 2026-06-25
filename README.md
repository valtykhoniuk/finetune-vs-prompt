# finetune-vs-prompt

Intent classification for FoxSchool support tickets. Compare **zero-shot** vs **few-shot** vs **LoRA fine-tune**.

**Labels:** `billing` · `bug` · `how-to` · `refund` · `other`

## Results (test set, n=50)

| Approach                       | Accuracy | Correct   | Cost\*             | Latency\*       |
| ------------------------------ | -------- | --------- | ------------------ | --------------- |
| Zero-shot (GPT-4o-mini API)    | 82%      | 41/50     | API per call       | API round-trip  |
| Few-shot (GPT-4o-mini API)     | 84%      | 42/50     | API, longer prompt | API round-trip  |
| **LoRA (Llama 3.2 1B, local)** | **86%**  | **43/50** | **$0 inference**   | local (MPS/CPU) |

\*Cost/latency not formally benchmarked; qualitative comparison for portfolio README.

LoRA wins on accuracy with no per-request API cost. The gain is modest (+2–4 pp) but consistent on the hardest class (`billing`).

### Per-label accuracy

| Label   | Zero-shot | Few-shot | LoRA    |
| ------- | --------- | -------- | ------- |
| refund  | 100%      | 100%     | 100%    |
| how-to  | 100%      | 100%     | 100%    |
| bug     | 90%       | 90%      | 90%     |
| other   | 90%       | 90%      | 80%     |
| billing | 36%       | 45%      | **64%** |

**Note:** `billing` vs `refund` / `how-to` is the weak zone for prompt-only approaches (e.g. pricing vs refund eligibility vs “how do I change plan”). LoRA improves `billing` the most (+19 pp vs zero-shot) but still confuses it with `bug` on edge cases.

## Conclusion

On 250 synthetic FoxSchool tickets (200 train / 50 test), **LoRA fine-tune beat both prompt baselines** (86% vs 84% few-shot vs 82% zero-shot). Prompt engineering is faster to iterate and good enough when API cost is acceptable; fine-tuning pays off when you need a **stable, cheap local classifier** and have ~200 labeled examples. For this dataset the margin is small — the bigger win is **billing accuracy** (+28 pp vs zero-shot), not a dramatic overall jump. If labels are scarce or the task changes often, start with few-shot; if you deploy at volume on open weights, LoRA is worth the Colab afternoon.

## LoRA setup

|            |                                                          |
| ---------- | -------------------------------------------------------- |
| Base model | `meta-llama/Llama-3.2-1B-Instruct`                       |
| Method     | LoRA (PEFT + TRL SFTTrainer), fp16 on Colab T4           |
| Train data | 200 tickets → `training/train_alpaca.jsonl`              |
| Adapter    | `models/foxschool-intent-lora/` (~45 MB, not in git)     |
| Eval       | `python inference/predict_lora.py` → `results/lora.json` |
| Notebook   | [`training/train_lora.ipynb`](training/train_lora.ipynb) |

## Dataset

| Split | Rows |
| ----- | ---- |
| Train | 200  |
| Test  | 50   |
| Total | 250  |

Synthetic tickets (GPT-generated) · split seed `42` · train/test held out for fair evaluation.

## Reproduce

```bash
# baselines (needs OPENAI_API_KEY in .env)
python baselines/zero_shot.py
python baselines/few_shot.py

# LoRA eval (needs huggingface-cli login for Llama)
pip install transformers peft accelerate torch
python inference/predict_lora.py
```
