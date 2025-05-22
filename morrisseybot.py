import json
from flask import Blueprint, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("smiths_lyrics_full_tagged.json", encoding="utf-8") as f:
    lyrics_data = json.load(f)

lyric_lines = [entry["line"] for entry in lyrics_data]
line_sources = [{"song": entry["song"], "album": entry["album"]} for entry in lyrics_data]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(lyric_lines)

morrissey_api = Blueprint("morrissey_api", __name__)

@morrissey_api.route("/api/morrissey", methods=["POST"])
def get_morrissey_reply():
    data = request.get_json()
    user_input = data.get("message", "")
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    query_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_index = similarity.argmax()

    print(f">>> MATCH: {lyric_lines[top_index]}")  # For Render log debug

    return jsonify({
        "reply": lyric_lines[top_index],
        "song": line_sources[top_index]["song"],
        "album": line_sources[top_index]["album"]
    })