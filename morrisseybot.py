import json
import os
import numpy as np
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai

# ✅ Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Create Blueprint BEFORE using it in route decorators
morrissey_api = Blueprint("morrissey_api", __name__)

# ✅ Load lyric chunks and embeddings
json_path = os.path.join(os.path.dirname(__file__), "smiths_lyrics_refined_chunks_embedded_cleaned.json")
with open(json_path, encoding="utf-8") as f:
    lyrics_data = json.load(f)

lyric_chunks = [entry["chunk"] for entry in lyrics_data]
line_sources = [{"song": entry["song"], "album": entry["album"]} for entry in lyrics_data]
embeddings = np.array([entry["embedding"] for entry in lyrics_data])

# ✅ Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ GPT-4 email generation function
def generate_email_gpt(user_input, lyric):
    prompt = f"""
You are Morrissey, responding to a fan's question with poetic melancholy, wit, and emotional distance.

Fan’s Question: “{user_input}”

Your Lyric: “{lyric}”

Write a short, characterful email reply as Morrissey. Use irony and sign off as Morrissey.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# ✅ Main route with logging and error capture
@morrissey_api.route("/api/morrissey", methods=["POST", "OPTIONS"])
@cross_origin(origin="https://morrisseybot-ui.vercel.app")
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
        print(">>> DEBUG: Top lyric chunk:", chunk)

        email = generate_email_gpt(user_input, chunk)
        print(">>> DEBUG: Email:", email)

        return jsonify({
            "reply": chunk,
            "song": line_sources[top_index]["song"],
            "album": line_sources[top_index]["album"],
            "email": email
        })

    except Exception as e:
        print(">>> ERROR in /api/morrissey:", str(e))
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500