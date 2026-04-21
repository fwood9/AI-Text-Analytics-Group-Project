import pandas as pd
from eval import evaluate_batch

# Check CSV loads correctly
df = pd.read_csv("dataset/test/labels.csv")

# Sanity check
gold = df["text"].tolist()

results = evaluate_batch(gold, gold)

print("\n=== Sanity Check: Gold vs Gold ===")
for metric, score in results.items():
    print(f"{metric}: {score:.4f}")