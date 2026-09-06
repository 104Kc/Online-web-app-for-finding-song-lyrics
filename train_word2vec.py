"""
Train a Word2Vec model on a Thai song-lyrics dataset and compute
one averaged vector per song ("song vector").

Same idea as the wongnai-gensim example (tokenize with PyThaiNLP,
train Word2Vec with gensim), just applied to song lyrics instead
of restaurant reviews, and with one extra step: averaging every
song's word vectors into a single vector so we can compare a
user's remembered lyric snippet against whole songs.

Usage:
    python train_word2vec.py --data data/lyrics_sample.csv --out output
"""
import argparse
import os

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from pythainlp.tokenize import word_tokenize


def tokenize_lyrics(text: str):
    """Tokenize Thai lyrics into words, dropping whitespace/punctuation."""
    tokens = word_tokenize(str(text), engine="newmm")
    return [t.strip() for t in tokens if t.strip()]


def build_song_vector(tokens, model) -> np.ndarray:
    """Average the Word2Vec vectors of every token found in the model's vocab."""
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def load_lyrics_dataset(path: str) -> pd.DataFrame:
    """Load the app's standard schema or the supplied Thai lyrics export."""
    df = pd.read_csv(path)

    if {"title", "artist", "lyrics"}.issubset(df.columns):
        return df[["song_id", "title", "artist", "lyrics"]].copy()

    lyrics_columns = [
        "field_lyrics_chorus",
        "field_lyrics_hook",
        "field_lyrics_lead",
    ]
    required_columns = {"title", "field_artis", *lyrics_columns}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    lyrics = df[lyrics_columns].fillna("").astype(str).agg("\n".join, axis=1)
    return pd.DataFrame(
        {
            "song_id": np.arange(1, len(df) + 1),
            "title": df["title"].fillna(""),
            "artist": df["field_artis"].fillna(""),
            "lyrics": lyrics,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/lyrics_sample.csv")
    parser.add_argument("--out", default="output")
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_lyrics_dataset(args.data)
    print(f"Loaded {len(df)} songs from {args.data}")

    print("Tokenizing lyrics with PyThaiNLP...")
    df["tokens"] = df["lyrics"].apply(tokenize_lyrics)

    print("Training Word2Vec (skip-gram)...")
    model = Word2Vec(
        sentences=df["tokens"].tolist(),
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=1,
        epochs=args.epochs,
    )

    print("Building one averaged vector per song...")
    song_vectors = np.stack(
        [build_song_vector(toks, model) for toks in df["tokens"]]
    )

    model.save(os.path.join(args.out, "lyrics_w2v.model"))
    np.save(os.path.join(args.out, "song_vectors.npy"), song_vectors)
    df[["song_id", "title", "artist", "lyrics"]].to_csv(
        os.path.join(args.out, "songs.csv"), index=False
    )

    print(f"Done. Model + vectors saved to {args.out}/")


if __name__ == "__main__":
    main()
