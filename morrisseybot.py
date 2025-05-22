# morrisseybot.py (using sentence embeddings)

import json
import os
import numpy as np
from flask import Blueprint, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-embedded lyric chunks
json_path = os.path.join(os.path.dirname(__file__), "smiths_lyrics_refined_chunks_embedded.json")
with open(json_path, encoding="utf-8") as f:
    lyrics_data = json.load(f)

# Extract chunks, metadata, and embeddings
lyric_chunks = [entry["chunk"] for entry in lyrics_data]
line_sources = [{"song": entry["song"], "album": entry["album"]} for entry in lyrics_data]
embeddings = np.array([entry["embedding"] for entry in lyrics_data])

# Load SentenceTransformer model for incoming queries
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up Flask Blueprint
morrissey_api = Blueprint("morrissey_api", __name__)

@morrissey_api.route("/api/morrissey", methods=["POST"])
def get_morrissey_reply():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Embed the user input
    query_vec = model.encode([user_input])
    similarity = cosine_similarity(query_vec, embeddings).flatten()
    top_index = similarity.argmax()

    # Debug output
    print(">>> EMBEDDING MATCH:", lyric_chunks[top_index])

    return jsonify({
        "reply": lyric_chunks[top_index],
        "song": line_sources[top_index]["song"],
        "album": line_sources[top_index]["album"]
    })