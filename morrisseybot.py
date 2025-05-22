from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FLATTEN TO INDIVIDUAL LINES
lyric_lines = []
line_sources = []

for song in songs:
    for line in song["lyrics"]:
        if len(line) > 10:
            lyric_lines.append(line)
            line_sources.append({
                "song": song["song_title"],
                "album": song["album"]
            })

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(lyric_lines)

@morrissey_api.route("/api/morrissey", methods=["POST"])
def get_morrissey_reply():
    data = request.get_json()
    user_input = data.get("message", "")
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # FIND BEST LINE MATCH
    query_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_index = similarity.argmax()

    return jsonify({
        "reply": lyric_lines[top_index],
        "song": line_sources[top_index]["song"],
        "album": line_sources[top_index]["album"]
    })