from flask import Flask, jsonify, render_template, request

from search import LyricsSearcher

app = Flask(__name__)
searcher = LyricsSearcher(model_dir="output")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"results": []})
    results = searcher.search(query, top_k=5)
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
