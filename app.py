import os
import csv
from functools import lru_cache

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=1)
def load_song_catalog():
    with open(os.path.join(BASE_DIR, "output", "songs.csv"), encoding="utf-8") as file:
        return list(csv.DictReader(file))


@lru_cache(maxsize=1)
def get_searcher():
    from search import LyricsSearcher

    return LyricsSearcher(model_dir=os.path.join(BASE_DIR, "output"))


def find_song(song_id):
    return next(
        (row for row in load_song_catalog() if int(row["song_id"]) == song_id),
        None,
    )


def normalize_lyrics(text):
    return "".join(character for character in text.casefold() if character.isalnum())


def lyric_similarity(query, lyrics):
    normalized_query = normalize_lyrics(query)
    normalized_lyrics = normalize_lyrics(lyrics)
    if len(normalized_query) < 3 or len(normalized_lyrics) < 3:
        return 0.0

    query_grams = {
        normalized_query[index : index + 2]
        for index in range(len(normalized_query) - 1)
    }
    lyric_grams = {
        normalized_lyrics[index : index + 2]
        for index in range(len(normalized_lyrics) - 1)
    }
    shared = len(query_grams & lyric_grams)
    recall = shared / len(query_grams)
    precision = shared / len(lyric_grams)
    return (2 * recall * precision) / (recall + precision) if shared else 0.0


def lyric_matches(query):
    matches = []
    for row in load_song_catalog():
        score = lyric_similarity(query, row["lyrics"])
        if score >= 0.12:
            matches.append(
                {
                    "song_id": int(row["song_id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "score": round(score, 4),
                }
            )
    return sorted(matches, key=lambda result: result["score"], reverse=True)[:5]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/song/<int:song_id>")
def song_page(song_id):
    row = find_song(song_id)
    if row is None:
        return render_template("song.html", song=None), 404

    song = {
        "song_id": int(row["song_id"]),
        "title": row["title"],
        "artist": row["artist"],
        "lyrics": row["lyrics"],
    }
    return render_template("song.html", song=song)


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"results": []})

    normalized_query = query.casefold()
    title_matches = [
        {
            "song_id": int(row["song_id"]),
            "title": row["title"],
            "artist": row["artist"],
            "score": 1.0,
        }
        for row in load_song_catalog()
        if normalized_query in row["title"].casefold()
    ]
    if title_matches:
        return jsonify({"results": title_matches[:5]})

    exact_lyric_matches = [
        {
            "song_id": int(row["song_id"]),
            "title": row["title"],
            "artist": row["artist"],
            "score": 0.99,
        }
        for row in load_song_catalog()
        if normalized_query in row["lyrics"].casefold()
    ]
    if exact_lyric_matches:
        return jsonify({"results": exact_lyric_matches[:5]})

    similar_lyric_matches = lyric_matches(query)
    if similar_lyric_matches:
        return jsonify({"results": similar_lyric_matches})

    try:
        results = get_searcher().search(query, top_k=5)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError):
        results = []
    return jsonify({"results": results})


@app.route("/api/songs/<int:song_id>")
def api_song(song_id):
    row = find_song(song_id)
    if row is None:
        return jsonify({"error": "Song not found"}), 404

    return jsonify(
        {
            "song_id": int(row["song_id"]),
            "title": row["title"],
            "artist": row["artist"],
            "lyrics": row["lyrics"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", "5000")))
