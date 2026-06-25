# finetune-vs-prompt

Intent classification for FoxSchool support tickets. Compare **zero-shot** vs **few-shot** vs **LoRA fine-tune**.

**Labels:** `billing` · `bug` · `how-to` · `refund` · `other`

## Dataset

| Split | Rows |
| ----- | ---- |
| Train | 200  |
| Test  | 50   |
| Total | 250  |

Synthetic tickets (GPT-generated) · split seed `42` · train/test held out for fair evaluation.

## Prompt baselines (test set, n=50)

| Approach  | Accuracy | Correct |
| --------- | -------- | ------- |
| Zero-shot | **82%**  | 41/50   |
| Few-shot  | **84%**  | 42/50   |

### Per-label accuracy

| Label   | Zero-shot | Few-shot |
| ------- | --------- | -------- |
| refund  | 100%      | 100%     |
| how-to  | 100%      | 100%     |
| bug     | 90%       | 90%      |
| other   | 90%       | 90%      |
| billing | 36%       | 45%      |

**Note:** `billing` vs `refund` / `how-to` is the weak zone for prompt-only approaches (e.g. pricing vs refund eligibility vs “how do I change plan”).

## Guides

- **Day 3–4 (LoRA, Colab, HF PEFT):** [`training/DAY3-4.md`](training/DAY3-4.md)
