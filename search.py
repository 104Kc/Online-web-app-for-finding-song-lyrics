"""
Core search logic: given a (possibly misremembered) lyric snippet,
find the most similar songs by cosine similarity between the
snippet's averaged Word2Vec vector and each song's averaged vector.

Because Word2Vec groups words that appear in similar contexts, this
tends to also work when a word is misspelled the way it was
misheard/misremembered -- as long as that misspelling shows up
somewhere in the training lyrics too (see README for how to make
that more robust).
"""
import os

import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize


def tokenize_lyrics(text: str):
    tokens = word_tokenize(str(text), engine="newmm")
    return [token.strip() for token in tokens if token.strip()]


class LyricsSearcher:
    def __init__(self, model_dir="output"):
        word_data = np.load(os.path.join(model_dir, "word_vectors.npz"))
        self.word_vectors = word_data["vectors"]
        self.vocabulary = word_data["vocabulary"]
        self.word_index = {
            str(word): index for index, word in enumerate(self.vocabulary)
        }
        self.song_vectors = np.load(os.path.join(model_dir, "song_vectors.npy"))
        self.songs = pd.read_csv(os.path.join(model_dir, "songs.csv"))

    def search(self, search_query: str, top_k: int = 5):
        search_query = search_query.strip()
        if not search_query:
            return []

        tokens = tokenize_lyrics(search_query)
        token_vectors = [
            self.word_vectors[self.word_index[token]]
            for token in tokens
            if token in self.word_index
        ]
        query_vec = (
            np.mean(token_vectors, axis=0)
            if token_vectors
            else np.zeros(self.word_vectors.shape[1])
        )
        song_norms = np.linalg.norm(self.song_vectors, axis=1)

        if np.any(query_vec):
            query_norm = np.linalg.norm(query_vec)
            denom = song_norms * query_norm
            denom[denom == 0] = 1e-9
            lyric_sims = (self.song_vectors @ query_vec) / denom
        else:
            lyric_sims = np.zeros(len(self.songs))

        normalized_query = search_query.casefold()
        title_sims = self.songs["title"].fillna("").astype(str).map(
            lambda title: 1.0 if normalized_query in title.casefold() else 0.0
        ).to_numpy()
        sims = np.maximum(lyric_sims, title_sims)

        if not np.any(sims):
            return []

        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for i in top_idx:
            row = self.songs.iloc[i]
            results.append(
                {
                    "song_id": int(row["song_id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "score": round(float(sims[i]), 4),
                }
            )
        return results


if __name__ == "__main__":
    # Quick manual test from the command line:
    #   python search.py "ฝนตกที่หน้าต่างบ้านฉัน"
    import sys

    cli_query = " ".join(sys.argv[1:]) or "ฝนตกที่หน้าต่าง"
    searcher = LyricsSearcher()
    for r in searcher.search(cli_query):
        print(f"{r['score']:.3f}  {r['title']} - {r['artist']}")
