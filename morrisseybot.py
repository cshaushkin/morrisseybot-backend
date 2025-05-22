import json
import os
import numpy as np
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load embedded chunks
json_path = os.path.join(os.path.dirname(__file__), "smiths_lyrics_refined_chunks_embedded_cleaned.json")
with open(json_path, encoding="utf-8") as f:
    lyrics_data = json.load(f)

# Extract data
lyric_chunks = [entry["chunk"] for entry in lyrics_data]
line_sources = [{"song": entry["song"], "album": entry["album"]} for entry in lyrics_data]
embeddings = np.array([entry["embedding"] for entry in lyrics_data])

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up API Blueprint
morrissey_api = Blueprint("morrissey_api", __name__)

@morrissey_api.route("/api/morrissey", methods=["POST", "OPTIONS"])
@cross_origin(origin="https://morrisseybot-ui.vercel.app")  # ✅ Enable CORS just for this route
def get_morrissey_reply():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Embed the query and match
    query_vec = model.encode([user_input])
    print(">>> DEBUG: query_vec shape:", np.array(query_vec).shape)
    print(">>> DEBUG: embeddings shape:", embeddings.shape)

    similarity = cosine_similarity(query_vec, embeddings).flatten()
    top_index = similarity.argmax()

    print(">>> DEBUG: best match index:", top_index)
    print(">>> DEBUG: selected chunk:", lyric_chunks[top_index])

    return jsonify({
        "reply": lyric_chunks[top_index],
        "song": line_sources[top_index]["song"],
        "album": line_sources[top_index]["album"]
    })