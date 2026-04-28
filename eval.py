# eval.py

import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

sbert = SentenceTransformer("all-MiniLM-L6-v2")


def parse_slots(sentence: str) -> dict:
    """
    Extract structured slots from a sentence.
    Returns: {num_dice, colours (list), values (list), sizes (list)}
    """
    sentence = sentence.lower()

    # Count dice
    count_map = {"one": 1, "two": 2, "three": 3}
    num_dice = None
    for word, n in count_map.items():
        if word in sentence:
            num_dice = n
            break

    # Extract colours
    colour_list = ["red", "blue", "green", "yellow", "white", "black",
                   "purple", "peach"]
    colours = [c for c in colour_list if c in sentence]

    # Extract values
    value_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    values = [n for w, n in value_map.items() if w in sentence]

    # Extract sizes
    size_list = ["small", "medium", "large"]
    sizes = [s for s in size_list if s in sentence]

    return {
        "num_dice": num_dice,
        "colours":  colours,
        "values":   sorted(values),
        "sizes":    sizes,
    }


def slot_accuracy(pred: str, gold: str) -> dict:
    """Per-slot accuracy between predicted and gold sentence."""
    p = parse_slots(pred)
    g = parse_slots(gold)
    count_correct   = int(p["num_dice"] == g["num_dice"])
    colours_correct = int(sorted(p["colours"]) == sorted(g["colours"]))
    values_correct  = int(p["values"] == g["values"])
    sizes_correct   = int(sorted(p["sizes"]) == sorted(g["sizes"]))
    fully_correct   = int(
        count_correct and colours_correct and values_correct and sizes_correct
    )
    return {
        "count_correct":   count_correct,
        "colours_correct": colours_correct,
        "values_correct":  values_correct,
        "sizes_correct":   sizes_correct,
        "fully_correct":   fully_correct,
    }


def cosine_sim(pred: str, gold: str) -> float:
    """SBERT cosine similarity between two sentences."""
    embs = sbert.encode([pred, gold])
    return float(cosine_similarity([embs[0]], [embs[1]])[0][0])


def bleu(pred: str, gold: str) -> float:
    """Sentence-level BLEU score."""
    sf = SmoothingFunction().method1
    return sentence_bleu([gold.split()], pred.split(), smoothing_function=sf)


def evaluate_batch(predictions: list, gold_labels: list) -> dict:
    """
    Evaluate a list of predictions against gold labels.
    Returns averaged metrics dict.
    """
    results = []
    for pred, gold in zip(predictions, gold_labels):
        slots = slot_accuracy(pred, gold)
        slots["cosine_sim"] = cosine_sim(pred, gold)
        slots["bleu"] = bleu(pred, gold)
        results.append(slots)

    # Average all metrics
    keys = results[0].keys()
    return {k: np.mean([r[k] for r in results]) for k in keys}
