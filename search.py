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
from gensim.models import Word2Vec

from train_word2vec import tokenize_lyrics, build_song_vector


class LyricsSearcher:
    def __init__(self, model_dir="output"):
        self.model = Word2Vec.load(os.path.join(model_dir, "lyrics_w2v.model"))
        self.song_vectors = np.load(os.path.join(model_dir, "song_vectors.npy"))
        self.songs = pd.read_csv(os.path.join(model_dir, "songs.csv"))

    def search(self, query: str, top_k: int = 5):
        tokens = tokenize_lyrics(query)
        query_vec = build_song_vector(tokens, self.model)

        if not np.any(query_vec):
            # None of the query's words were in the training vocabulary
            return []

        song_norms = np.linalg.norm(self.song_vectors, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denom = song_norms * query_norm
        denom[denom == 0] = 1e-9

        sims = (self.song_vectors @ query_vec) / denom

        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for i in top_idx:
            row = self.songs.iloc[i]
            results.append(
                {
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

    query = " ".join(sys.argv[1:]) or "ฝนตกที่หน้าต่าง"
    searcher = LyricsSearcher()
    for r in searcher.search(query):
        print(f"{r['score']:.3f}  {r['title']} - {r['artist']}")
