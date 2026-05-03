# LLM baseline comparison using GPT-4o Vision

import openai
import base64
import pandas as pd
from eval import evaluate_batch

client = openai.OpenAI(api_key="OPENAI_API_KEY")

DATASET_ROOT = "dataset"
TEST_CSV = f"{DATASET_ROOT}/test/labels.csv"
TEST_IMGS = f"{DATASET_ROOT}/test/images"

ZERO_SHOT_PROMPT = """Describe this image of dice.
State: how many dice, their colours, their sizes (small/medium/large), 
and the face value shown on each die.
Use this format: 'There are [N] [size] [colour] dice showing [values].'"""

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def gpt4o_predict_zero_shot(image_path: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image(image_path)}"
                    }
                },
                {"type": "text", "text": ZERO_SHOT_PROMPT}
            ]
        }]
    )
    return response.choices[0].message.content.strip()

def gpt4o_predict_one_shot(image_path: str, example_caption: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Example: '{example_caption}'\n\nNow describe this image in the same format:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image(image_path)}"
                    }
                },
                {"type": "text", "text": ZERO_SHOT_PROMPT}
            ]
        }]
    )
    return response.choices[0].message.content.strip()

def run_comparison(n_samples: int = 100):
    df = pd.read_csv(TEST_CSV).head(n_samples).reset_index(drop=True)
    gold = df["text"].tolist()
    example_caption = gold[0]

    preds_zero, preds_one = [], []

    for i, row in df.iterrows():
        img_path = f"{TEST_IMGS}/{row['image']}"
        preds_zero.append(gpt4o_predict_zero_shot(img_path))
        preds_one.append(gpt4o_predict_one_shot(img_path, example_caption))
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{n_samples}")

    metrics_zero = evaluate_batch(preds_zero, gold)
    metrics_one  = evaluate_batch(preds_one,  gold)

    print("\n=== Zero-shot GPT-4o ===")
    for k, v in metrics_zero.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== One-shot GPT-4o ===")
    for k, v in metrics_one.items():
        print(f"  {k}: {v:.4f}")

    df["pred_zero_shot"] = preds_zero
    df["pred_one_shot"]  = preds_one
    df.to_csv("llm_predictions.csv", index=False)
    print("\nSaved to llm_predictions.csv")

    return metrics_zero, metrics_one

if __name__ == "__main__":
    run_comparison(n_samples=100)