# morrisseybot.py

import json
import os
from flask import Blueprint, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load flat line-by-line JSON from the same directory as this file
json_path = os.path.join(os.path.dirname(__file__), "smiths_lyrics_full_tagged.json")
with open(json_path, encoding="utf-8") as f:
    lyrics_data = json.load(f)

# Build list of individual lyric lines and their metadata
lyric_lines = [entry["line"] for entry in lyrics_data]
line_sources = [{"song": entry["song"], "album": entry["album"]} for entry in lyrics_data]

# Build TF-IDF matrix for lyric lines
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(lyric_lines)

# Set up Flask Blueprint
morrissey_api = Blueprint("morrissey_api", __name__)

@morrissey_api.route("/api/morrissey", methods=["POST"])
def get_morrissey_reply():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # TF-IDF match
    query_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_index = similarity.argmax()

    # DEBUG: print first few lines so you can verify it's working
    print(">>> DEBUG: First few lyric lines loaded:")
    for i in range(3):
        print(f"{i+1}. {lyric_lines[i]}")

    print(f">>> DEBUG: Selected match: {lyric_lines[top_index]}")

    return jsonify({
        "reply": lyric_lines[top_index],
        "song": line_sources[top_index]["song"],
        "album": line_sources[top_index]["album"]
    })