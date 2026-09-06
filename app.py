from flask import Flask, jsonify, render_template, request

from search import LyricsSearcher

app = Flask(__name__)
searcher = LyricsSearcher(model_dir="output")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/song/<int:song_id>")
def song_page(song_id):
    matches = searcher.songs[searcher.songs["song_id"] == song_id]
    if matches.empty:
        return render_template("song.html", song=None), 404

    row = matches.iloc[0]
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
    results = searcher.search(query, top_k=5)
    return jsonify({"results": results})


@app.route("/api/songs/<int:song_id>")
def api_song(song_id):
    matches = searcher.songs[searcher.songs["song_id"] == song_id]
    if matches.empty:
        return jsonify({"error": "Song not found"}), 404

    row = matches.iloc[0]
    return jsonify(
        {
            "song_id": int(row["song_id"]),
            "title": row["title"],
            "artist": row["artist"],
            "lyrics": row["lyrics"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
