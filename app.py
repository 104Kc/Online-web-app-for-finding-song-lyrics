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

    lyric_matches = [
        {
            "song_id": int(row["song_id"]),
            "title": row["title"],
            "artist": row["artist"],
            "score": 0.99,
        }
        for row in load_song_catalog()
        if normalized_query in row["lyrics"].casefold()
    ]
    if lyric_matches:
        return jsonify({"results": lyric_matches[:5]})

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
