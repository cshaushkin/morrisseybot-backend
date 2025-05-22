# morrisseybot.py

import json
import os
from flask import Blueprint, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load chunked lyric file
json_path = os.path.join(os.path.dirname(__file__), "smiths_lyrics_chunks.json")
with open(json_path, encoding="utf-8") as f:
    lyrics_data = json.load(f)

# Build lyric chunks and associated metadata
lyric_chunks = [entry["chunk"] for entry in lyrics_data]
line_sources = [{"song": entry["song"], "album": entry["album"]} for entry in lyrics_data]

# TF-IDF vectorization on lyric chunks
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(lyric_chunks)

# Set up Flask Blueprint
morrissey_api = Blueprint("morrissey_api", __name__)

@morrissey_api.route("/api/morrissey", methods=["POST"])
def get_morrissey_reply():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Find best-matching lyric chunk
    query_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_index = similarity.argmax()

    print(">>> DEBUG: Selected chunk:")
    print(lyric_chunks[top_index])

    return jsonify({
        "reply": lyric_chunks[top_index],
        "song": line_sources[top_index]["song"],
        "album": line_sources[top_index]["album"]
    })