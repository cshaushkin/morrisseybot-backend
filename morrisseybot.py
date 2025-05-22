import json
import os
import numpy as np
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai

# Load OpenAI key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load embedded chunk data
json_path = os.path.join(os.path.dirname(__file__), "smiths_lyrics_refined_chunks_embedded_cleaned.json")
with open(json_path, encoding="utf-8") as f:
    lyrics_data = json.load(f)

# Extract lyric chunks and metadata
lyric_chunks = [entry["chunk"] for entry in lyrics_data]
line_sources = [{"song": entry["song"], "album": entry["album"]} for entry in lyrics_data]
embeddings = np.array([entry["embedding"] for entry in lyrics_data])

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up the Flask blueprint
morrissey_api = Blueprint("morrissey_api", __name__)

# Generate a short email in Morrissey's voice using GPT-4
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

# Define the API route
@morrissey_api.route("/api/morrissey", methods=["POST", "OPTIONS"])
@cross_origin(origin="https://morrisseybot-ui.vercel.app")
def get_morrissey_reply():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    query_vec = model.encode([user_input])
    print(">>> DEBUG: query_vec shape:", np.array(query_vec).shape)
    print(">>> DEBUG: embeddings shape:", embeddings.shape)

    similarity = cosine_similarity(query_vec, embeddings).flatten()
    top_index = similarity.argmax()

    chunk = lyric_chunks[top_index]
    print(">>> DEBUG: selected chunk:", chunk)

    email = generate_email_gpt(user_input, chunk)

    return jsonify({
        "reply": chunk,
        "song": line_sources[top_index]["song"],
        "album": line_sources[top_index]["album"],
        "email": email
    })