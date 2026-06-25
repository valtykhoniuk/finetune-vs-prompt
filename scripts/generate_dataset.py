import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
LABELS = ["billing", "bug", "how-to", "refund", "other"]
PER_LABEL = 50
BATCH_SIZE = 10
OUTPUT = Path("data/tickets_labeled.jsonl")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM = """You generate realistic FoxSchool support tickets.
FoxSchool is an online language learning SaaS (Beginner/Intermediate/Pro plans,
video lessons, PDF worksheets, subscriptions).
Output ONLY a JSON array of strings — customer messages, no labels inside text.
Each message 1-2 sentences, natural English, varied wording.
Do NOT repeat the same opening phrase."""

def generate_batch(label: str, n: int) -> list[str]:
    prompt = f"""Generate {n} different customer support messages for intent: {label}.
billing = payment, charges, invoices, plan price
bug = something broken, crash, error, not loading
how-to = how do I... questions about using the product
refund = want money back, refund eligibility
other = off-topic, greetings, thanks, unrelated questions
Return JSON array of {n} strings only."""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.9,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    raw = resp.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    return json.loads(raw)

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    seen = set()

    with OUTPUT.open("w", encoding="utf-8") as f:
        for label in LABELS:
            remaining = PER_LABEL
            while remaining > 0:
                batch_n = min(BATCH_SIZE, remaining)
                texts = generate_batch(label, batch_n)
                for text in texts:
                    text = text.strip()
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    row = {"text": text, "label": label}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                remaining -= len(texts)
                print(f"{label}: wrote batch, remaining ~{remaining}")
                time.sleep(0.5)  
    print(f"Done: {OUTPUT}")

if __name__ == "__main__":
    main()
