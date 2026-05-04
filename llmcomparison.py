import os, base64, pandas as pd, time
from pathlib import Path
from google import genai
from google.genai import types
from eval import evaluate_batch

client = genai.Client(api_key="your_gemini_api_key")

DATASET_ROOT = "dataset"
TEST_CSV  = Path(DATASET_ROOT) / "test" / "labels.csv"
TEST_IMGS = Path(DATASET_ROOT) / "test" / "images"
MODEL     = "gemini-2.5-flash"

ZERO_SHOT_PROMPT = (
    "Describe this image of dice. State: how many dice, their colours, "
    "their sizes (small/medium/large), and the face value shown on each die. "
    "Use this format: 'There are [N] [size] [colour] dice showing [values].' "
    "or for mixed colours: 'The image shows a [size] [colour] die showing [value] "
    "and a [size] [colour] die showing [value].'"
)

def gemini_predict(image_path, prompt, example_caption=None):
    if example_caption:
        full_prompt = f"Example: '{example_caption}'\nNow describe this image in the same format:\n{prompt}"
    else:
        full_prompt = prompt

    for attempt in range(5):  # retry up to 5 times
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(
                        data=open(image_path, "rb").read(),
                        mime_type="image/png"
                    ),
                    full_prompt
                ]
            )
            return response.text.strip()
        except Exception as e:
            wait = 60 * (attempt + 1)  # 60s, 120s, 180s...
            print(f"  Rate limit hit, waiting {wait}s... (attempt {attempt+1}/5)")
            time.sleep(wait)

    return ""  # return empty string if all retries fail

def run_comparison(n_samples=5):
    df = pd.read_csv(TEST_CSV).head(n_samples).reset_index(drop=True)
    gold = df["text"].tolist()
    example_caption = gold[0]

    preds_zero, preds_one = [], []

    for i, row in df.iterrows():
        img_path = TEST_IMGS / row["image"]

        print(f"[{i+1}/{n_samples}] Zero-shot...")
        preds_zero.append(gemini_predict(img_path, ZERO_SHOT_PROMPT))
        time.sleep(15)

        print(f"[{i+1}/{n_samples}] One-shot...")
        preds_one.append(gemini_predict(img_path, ZERO_SHOT_PROMPT, example_caption))
        time.sleep(15)

    metrics_zero = evaluate_batch(preds_zero, gold)
    metrics_one  = evaluate_batch(preds_one,  gold)

    print("\n=== Zero-shot Gemini 2.5 Flash ===")
    for k, v in metrics_zero.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== One-shot Gemini 2.5 Flash ===")
    for k, v in metrics_one.items():
        print(f"  {k}: {v:.4f}")

    Path("outputs").mkdir(exist_ok=True)
    df["pred_zero_shot"] = preds_zero
    df["pred_one_shot"]  = preds_one
    df.to_csv("outputs/llm_predictions.csv", index=False)
    print("\nSaved to outputs/llm_predictions.csv")

    return metrics_zero, metrics_one

if __name__ == "__main__":
    run_comparison(n_samples=5)