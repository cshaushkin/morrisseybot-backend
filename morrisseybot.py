import json
import os
import numpy as np
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ✅ Create Flask Blueprint
morrissey_api = Blueprint("morrissey_api", __name__)

# ✅ Load lyric chunks + embeddings
json_path = os.path.join(os.path.dirname(__file__), "smiths_lyrics_refined_chunks_embedded_cleaned.json")
with open(json_path, encoding="utf-8") as f:
    lyrics_data = json.load(f)

lyric_chunks = [entry["chunk"] for entry in lyrics_data]
line_sources = [{"song": entry["song"], "album": entry["album"]} for entry in lyrics_data]
embeddings = np.array([entry["embedding"] for entry in lyrics_data])

# ✅ Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Free mock email generator
def generate_email_gpt(user_input, lyric):
    return f"""Dear friend,

Your question — “{user_input}” — reminded me of something I once sang:

    “{lyric}”

I hope this clarifies nothing at all.

Yours (begrudgingly),
Morrissey"""

# ✅ MorrisseyBot API Endpoint
@morrissey_api.route("/api/morrissey", methods=["POST", "OPTIONS"])
@cross_origin(
    origin="https://morrisseybot-ui.vercel.app",
    methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)
def get_morrissey_reply():
    try:
        print(">>> DEBUG: Received POST /api/morrissey")

        data = request.get_json(force=True)
        user_input = data.get("message", "")
        print(">>> DEBUG: User input:", user_input)

        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        query_vec = model.encode([user_input])
        print(">>> DEBUG: Query vector shape:", query_vec.shape)

        similarity = cosine_similarity(query_vec, embeddings).flatten()
        top_index = similarity.argmax()

        chunk = lyric_chunks[top_index]
        print(">>> DEBUG: Matched chunk:", chunk)

        email = generate_email_gpt(user_input, chunk)
        print(">>> DEBUG: Mock email generated")

        return jsonify({
            "reply": chunk,
            "song": line_sources[top_index]["song"],
            "album": line_sources[top_index]["album"],
            "email": email
        })

    except Exception as e:
        print(">>> ERROR in /api/morrissey:", str(e))
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500