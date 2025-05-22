@morrissey_api.route("/api/morrissey", methods=["POST", "OPTIONS"])
@cross_origin(origin="https://morrisseybot-ui.vercel.app")
def get_morrissey_reply():
    try:
        print(">>> DEBUG: Received POST /api/morrissey")

        # Force parse JSON from request body
        data = request.get_json(force=True)
        user_input = data.get("message", "")

        print(">>> DEBUG: User input:", user_input)

        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        # Embed the user query
        query_vec = model.encode([user_input])
        print(">>> DEBUG: Query vector shape:", query_vec.shape)

        # Compare with stored embeddings
        similarity = cosine_similarity(query_vec, embeddings).flatten()
        top_index = similarity.argmax()

        chunk = lyric_chunks[top_index]
        print(">>> DEBUG: Matched chunk:", chunk)

        # Generate email from GPT
        email = generate_email_gpt(user_input, chunk)
        print(">>> DEBUG: GPT Email:", email)

        return jsonify({
            "reply": chunk,
            "song": line_sources[top_index]["song"],
            "album": line_sources[top_index]["album"],
            "email": email
        })

    except Exception as e:
        # Catch and log the full error
        print(">>> ERROR in /api/morrissey:", str(e))
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500