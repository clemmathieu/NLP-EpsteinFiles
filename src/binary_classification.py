"""
Stage 1 - Binary Classification
================================
Loads the Epstein emails dataset from Hugging Face, preprocesses every
thread, then uses ``facebook/bart-large-mnli`` (zero-shot NLI) to assign
a probability score to each thread and flag it as *problematic* when that
score exceeds BINARY_THRESHOLD.

Architecture
------------
* Model   : facebook/bart-large-mnli  (zero-shot NLI - no fine-tuning needed)
* Labels  : ['problematic', 'non-problematic']
* Threshold: 0.35  - deliberately low to favour recall over precision;
  analysts can review flagged emails but cannot review what was never flagged.

Outputs
-------
data/binary_classified.csv
"""

import json
import os
import re

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline

# Parameters
BINARY_LABELS    = ["problematic", "non-problematic"]
BINARY_THRESHOLD = 0.35     # flag if P(problematic) >= 35 %
MAX_TEXT_LEN     = 512      # chars fed to the classifier per thread
DEMO_MODE        = True     # False = on the full dataset
DEMO_SIZE        = 800
OUTPUT_PATH      = "data/binary_classified.csv"


def parse_messages(messages_field) -> tuple:
    """
    Takes a thread's raw 'messages' field (string or list) and extracts:
    (full_text: str, unique_senders: list[str], date_range: str, msg_count: int)
    """
    try:
        # convert JSON strings or python lists to Python objects
        if isinstance(messages_field, str):
            messages = json.loads(messages_field)
        elif isinstance(messages_field, list):
            messages = messages_field
        else:
            return "", [], "Unknown", 0

        # lists to store extracted info for each message
        bodies, senders, dates = [], [], []

        for msg in messages:
            # the ors check to handle different possible field names and missing data
            body   = msg.get("body", "") or msg.get("content", "") or ""
            sender = msg.get("sender", "") or msg.get("from", "") or ""
            date   = msg.get("date", "") or msg.get("timestamp", "") or ""

            if body: bodies.append(str(body).strip())
            if sender: senders.append(str(sender).strip())
            if date: dates.append(str(date).strip())

        # all message contents in the thread are merged into one string with [MSG] separators
        full_text = "[MSG]".join(bodies)
        unique_senders = list(dict.fromkeys(senders))
        date_range = (
            f"{dates[0]} → {dates[-1]}" if len(dates) >= 2
            else (dates[0] if dates else "Unknown")
        )
        return full_text, unique_senders, date_range, len(messages)

    except Exception:
        return "", [], "Unknown", 0


def clean_text(text: str) -> str:
    """Cleans the text by removing HTML tags, headers, and non-printable characters."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(From|To|Subject|Date|Cc|Bcc):\s*[^\n]+\n", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E]", " ", text)
    return text.strip()


def load_and_preprocess() -> pd.DataFrame:
    """
    Download the dataset from Hugging Face, parse every thread's messages, clean the text, and return a DataFrame ready for classification.
    """
    print("Downloading dataset from Hugging Face...")
    raw = load_dataset("notesbymuneeb/epstein-emails")
    df  = raw["train"].to_pandas()
    print(f"Loaded {len(df):,} rows | columns: {list(df.columns)}")

    print("Parsing messages and cleaning text")
    #For each row, run the parser
    parsed = df["messages"].apply(parse_messages)

    df["full_text"]     = [clean_text(p[0]) for p in parsed]
    df["senders"]       = [p[1] for p in parsed]
    df["date_range"]    = [p[2] for p in parsed]
    df["message_count"] = [p[3] for p in parsed]

    # Ensure message_count exists even if the dataset already had one
    if "message_count" not in df.columns or df["message_count"].isna().all():
        df["message_count"] = [p[3] for p in parsed]
    
    # fills missing subjects to avoid null values
    df["subject"] = df["subject"].fillna("[No Subject]")

    # so that the model classifies subject + body text
    df["text_for_classification"] = (df["subject"] + " " + df["full_text"]).str.strip()

    # Ensure thread_id exists
    if "thread_id" not in df.columns: df["thread_id"] = df.index.astype(str)

    before = len(df)
    #remove very short threads
    df = df[df["full_text"].str.len() > 20].reset_index(drop=True)
    print(f"  Removed {before - len(df)} empty / very-short threads")
    print(f"  ✅ {len(df):,} valid threads remaining")
    return df


def _extract_score(result: dict, label: str) -> float:
    '''Extracts the score for a given label from the zero-shot classification result dict.'''
    try:
        return result["scores"][result["labels"].index(label)]
    # will return 0.0 if the label is not found or if the expected keys are missing
    except Exception:
        return 0.0


def classify_binary_batch(
    texts: list,
    zero_shot,
    batch_size: int = 16,
) -> list:
    """
    Run zero-shot binary classification in batches instead of one thread at a time.
    Returns a list of raw pipeline result dicts.
    """
    results     = []
    #keep only the first 512 characters of each thread to keep computation manageable
    texts_trunc = [t[:MAX_TEXT_LEN] if t else "empty email" for t in texts]

    for i in tqdm(range(0, len(texts_trunc), batch_size), desc="Binary classification"):
        batch = texts_trunc[i : i + batch_size]
        try:
            # multi_label = False means the model must choose between the two labels
            out = zero_shot(batch, BINARY_LABELS, multi_label=False)
            results.extend(out if isinstance(out, list) else [out])
        except Exception:
            results.extend(
                [{"labels": ["non-problematic"], "scores": [1.0]}] * len(batch)
            )
    return results


def run(demo_mode: bool = DEMO_MODE) -> pd.DataFrame:
    """
    Execute the full Stage-1 pipeline.

    Parameters
    demo_mode : bool
        When True, only classify the first `DEMO_SIZE` threads.

    Returns
    pd.DataFrame
        DataFrame with `prob_score` and `risk_flag` columns appended,
        also saved to `OUTPUT_PATH`.
    """
    df = load_and_preprocess()

    if demo_mode:
        classify_df = df.head(DEMO_SIZE).copy()
        print(f"⚠️  DEMO MODE - classifying first {DEMO_SIZE} threads only")
    else:
        classify_df = df.copy()
        print(f"Running on full dataset ({len(df):,} threads) ...")

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading classifier on {'GPU' if device == 0 else 'CPU'} ...")
    zero_shot = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )

    binary_results = classify_binary_batch(
        classify_df["text_for_classification"].tolist(), zero_shot
    )

    #stores probability that each thread is 'problematic'
    classify_df["prob_score"] = [
        _extract_score(r, "problematic") for r in binary_results
    ]
    classify_df["risk_flag"] = (
        classify_df["prob_score"] >= BINARY_THRESHOLD
    ).astype(int)

    n_flagged = int(classify_df["risk_flag"].sum())
    pct       = 100 * n_flagged / len(classify_df)
    print(f"\nStage 1 complete")
    print(f"Flagged as Problematic : {n_flagged} ({pct:.1f}%)")
    print(f"Flagged as Non-Problematic : {len(classify_df) - n_flagged} ({100 - pct:.1f}%)")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    save_df = classify_df.copy()
    save_df["senders"] = save_df["senders"].apply(
        lambda x: str(x) if isinstance(x, list) else x
    )
    save_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

    return classify_df


if __name__ == "__main__":
    run()