"""
Stage 2 - Offense Classification  +  NER  +  Sentence Embeddings
=================================================================
Takes the Stage-1 output (``data/binary_classified.csv``), and for every
*flagged* thread applies multi-label zero-shot classification across six
offense categories.  Named-entity recognition and sentence embeddings are
then computed for the full dataset, and the final artefacts are saved.

Architecture
------------
* Classifier : facebook/bart-large-mnli  (multi_label=True)
* Categories : six taxonomy labels derived from known Epstein case facts
* Threshold  : 0.30 per label - intentionally permissive for recall
* NER        : spaCy en_core_web_sm
* Embeddings : all-MiniLM-L6-v2  (L2-normalised → cosine via inner product)

Outputs
-------
data/classified_emails.csv  -  final annotated DataFrame (consumed by app.py)
data/embeddings.npy          -  float32 array of shape (n_threads, 384)
"""

import ast
import os

import numpy as np
import pandas as pd
import spacy
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import pipeline

# ── Configuration ─────────────────────────────────────────────────────────────
OFFENSE_CATEGORIES = [
    "sexual exploitation or trafficking",
    "financial fraud or money laundering",
    "obstruction or witness tampering",
    "bribery or corruption",
    "coercion or blackmail",
    "network facilitation or coordination",
]

OFFENSE_THRESHOLD  = 0.30          # apply label when score >= threshold
MAX_TEXT_LEN       = 512
EMBEDDER_MODEL     = "all-MiniLM-L6-v2"
INPUT_PATH         = "data/binary_classified.csv"
OUTPUT_CSV         = "data/classified_emails.csv"
OUTPUT_EMBEDDINGS  = "data/embeddings.npy"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_eval_list(value) -> list:
    """Safely deserialise a stringified Python list back to a real list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            result = ast.literal_eval(value)
            return result if isinstance(result, list) else [value]
        except Exception:
            return [value] if value else []
    return []


def _extract_score(result: dict, label: str) -> float:
    try:
        return result["scores"][result["labels"].index(label)]
    except Exception:
        return 0.0


# ── Offense classification ────────────────────────────────────────────────────

def classify_offenses_batch(
    texts: list,
    zero_shot,
    batch_size: int = 8,
) -> list:
    """
    Run multi-label offense classification in batches.
    Returns a list of raw pipeline result dicts.
    """
    results     = []
    texts_trunc = [t[:MAX_TEXT_LEN] if t else "empty" for t in texts]

    for i in tqdm(range(0, len(texts_trunc), batch_size), desc="Offense classification"):
        batch = texts_trunc[i : i + batch_size]
        try:
            out = zero_shot(batch, OFFENSE_CATEGORIES, multi_label=True)
            results.extend(out if isinstance(out, list) else [out])
        except Exception:
            results.extend(
                [{"labels": OFFENSE_CATEGORIES,
                  "scores": [0.0] * len(OFFENSE_CATEGORIES)}] * len(batch)
            )
    return results


def _get_offense_labels(row: pd.Series) -> list:
    """Return all offense labels that exceed the threshold for a given row."""
    labels = [
        cat for cat in OFFENSE_CATEGORIES
        if row.get("score_" + cat.replace(" ", "_").replace("/", "_"), 0.0)
        >= OFFENSE_THRESHOLD
    ]
    return labels if labels else ["unclassified"]


# ── NER ───────────────────────────────────────────────────────────────────────

def extract_entities(text: str, nlp) -> dict:
    """
    Extract named entities (PERSON, ORG, GPE, LOC, DATE, MONEY) from text
    using a pre-loaded spaCy pipeline.
    """
    if not text or len(text) < 10:
        return {}
    doc      = nlp(text[:1500])
    entities: dict = {}
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY"}:
            bucket = entities.setdefault(ent.label_, [])
            name   = ent.text.strip()
            if name not in bucket:
                bucket.append(name)
    return entities


# ── Sentence embeddings ───────────────────────────────────────────────────────

def build_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Encode subject + first 200 chars of body for every thread.
    Returns a float32 array of shape (n, 384), L2-normalised for cosine search.
    """
    print("Loading sentence-transformer model …")
    embedder = SentenceTransformer(EMBEDDER_MODEL)

    subjects = df["subject"].fillna("").astype(str)
    full_texts = df["full_text"].fillna("").astype(str).str[:200]

    texts = (subjects + " " + full_texts).tolist()
    print(f"Encoding {len(texts):,} threads …")

    emb = embedder.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype("float32")
    return emb


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    """
    Execute the full Stage-2 pipeline.

    Reads  : data/binary_classified.csv
    Writes : data/classified_emails.csv
             data/embeddings.npy

    Returns
    -------
    pd.DataFrame  -  final annotated DataFrame
    """
    print(f"Loading Stage-1 output from {INPUT_PATH} …")
    classify_df = pd.read_csv(INPUT_PATH)

    # Restore list columns serialised by Stage 1
    classify_df["senders"] = classify_df["senders"].apply(_safe_eval_list)

    # ── Stage-2 zero-shot classifier ──────────────────────────────────────────
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading classifier on {'GPU' if device == 0 else 'CPU'} …")
    zero_shot = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )

    flagged_mask = classify_df["risk_flag"] == 1
    flagged_df   = classify_df[flagged_mask].copy()
    print(f"Running Stage-2 on {len(flagged_df)} flagged emails …")

    offense_results = classify_offenses_batch(
        flagged_df["text_for_classification"].tolist(), zero_shot
    )

    # Write per-category scores back to flagged_df
    for cat in OFFENSE_CATEGORIES:
        col = "score_" + cat.replace(" ", "_").replace("/", "_")
        flagged_df[col] = [_extract_score(r, cat) for r in offense_results]

    flagged_df["offense_labels"]  = flagged_df.apply(_get_offense_labels, axis=1)
    flagged_df["primary_offense"] = flagged_df["offense_labels"].apply(lambda x: x[0])

    # Initialise columns for the full dataframe (non-flagged emails get "none")
    for cat in OFFENSE_CATEGORIES:
        col = "score_" + cat.replace(" ", "_").replace("/", "_")
        classify_df[col] = 0.0
    classify_df["offense_labels"]  = [["none"]] * len(classify_df)
    classify_df["primary_offense"] = "none"

    # Merge flagged results back in
    for idx, row in flagged_df.iterrows():
        classify_df.at[idx, "offense_labels"]  = row["offense_labels"]
        classify_df.at[idx, "primary_offense"] = row["primary_offense"]
        for cat in OFFENSE_CATEGORIES:
            col = "score_" + cat.replace(" ", "_").replace("/", "_")
            classify_df.at[idx, col] = row[col]

    print("\n✅ Stage-2 complete")
    offense_counts: dict = {}
    for labels in flagged_df["offense_labels"]:
        for lab in labels:
            offense_counts[lab] = offense_counts.get(lab, 0) + 1
    for k, v in sorted(offense_counts.items(), key=lambda x: -x[1]):
        print(f"   {k}: {v}")

    # ── NER ───────────────────────────────────────────────────────────────────
    print("\nRunning NER on all email threads …")
    nlp = spacy.load("en_core_web_sm")
    classify_df["entities"] = classify_df["full_text"].apply(
        lambda t: extract_entities(t, nlp)
    )

    top_persons = (
        pd.Series(
            [p for ents in classify_df[flagged_mask]["entities"]
             for p in ents.get("PERSON", [])]
        ).value_counts().head(20)
    )
    print("\nTop persons in flagged emails:")
    print(top_persons.head(10).to_string())

    # ── Embeddings ────────────────────────────────────────────────────────────
    embeddings = build_embeddings(classify_df)

    # ── Save artefacts ────────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)

    # Add display-friendly string columns (easier for the app to consume)
    classify_df["offense_labels_display"] = classify_df["offense_labels"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )
    classify_df["senders_display"] = classify_df["senders"].apply(
        lambda x: ", ".join(x[:3]) if isinstance(x, list) else str(x)
    )
    classify_df["risk_display"] = classify_df["risk_flag"].map(
        {1: "⚠️ Problematic", 0: "✅ Safe"}
    )
    classify_df["entities_json"] = classify_df["entities"].apply(str)

    # Serialise list columns so CSV round-trips cleanly
    save_df = classify_df.copy()
    save_df["senders"]       = save_df["senders"].apply(str)
    save_df["offense_labels"] = save_df["offense_labels"].apply(str)
    save_df["entities"]      = save_df["entities"].apply(str)

    save_df.to_csv(OUTPUT_CSV, index=False)
    np.save(OUTPUT_EMBEDDINGS, embeddings)

    print(f"\n✅ Pipeline complete")
    print(f"   CSV        → {OUTPUT_CSV}  ({len(save_df):,} rows)")
    print(f"   Embeddings → {OUTPUT_EMBEDDINGS}  {embeddings.shape}")

    return classify_df


if __name__ == "__main__":
    run()
