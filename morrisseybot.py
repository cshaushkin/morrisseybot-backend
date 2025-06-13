import json
import os
import numpy as np
import requests
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

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/gpt2"
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # You'll need to set this in your environment

def generate_morrissey_style_email(user_input, lyric):
    # Create a more detailed prompt that better captures Morrissey's style
    prompt = f"""Morrissey is writing an email response to a fan's question. His writing style is melancholic, witty, and often self-deprecating. He frequently references his own lyrics and uses dramatic language.

Fan's question: "{user_input}"

Morrissey's response:

Dear friend,

Your question — "{user_input}" — reminded me of something I once sang:

    "{lyric}"

*Morrissey continues writing in his characteristic style, expressing his thoughts about the question while maintaining his melancholic and witty tone. He might reference other songs or express his disdain for modern life. He should end with his signature sign-off.*

Yours (begrudgingly),
Morrissey"""

    # Call Hugging Face API with adjusted parameters
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 500,  # Increased for more context
            "num_return_sequences": 1,
            "no_repeat_ngram_size": 3,  # Increased to avoid repetition
            "do_sample": True,
            "top_k": 40,  # Slightly reduced for more focused output
            "top_p": 0.92,  # Slightly reduced for more focused output
            "temperature": 0.85,  # Increased for more creative responses
            "repetition_penalty": 1.2,  # Added to reduce repetition
            "length_penalty": 1.0,  # Added to encourage longer responses
        }
    }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        # Extract and clean the generated text
        generated_text = response.json()[0]["generated_text"]
        
        # Extract only the response part (after the prompt)
        try:
            # Find where the actual response starts (after the prompt)
            response_start = generated_text.find("Dear friend,")
            if response_start != -1:
                generated_text = generated_text[response_start:]
            
            # Clean up any extra newlines and ensure proper formatting
            generated_text = "\n".join(line.strip() for line in generated_text.split("\n") if line.strip())
            
            # Ensure we have a proper signature
            if "Yours" not in generated_text:
                generated_text += "\n\nYours (begrudgingly),\nMorrissey"
            elif not generated_text.endswith("Morrissey"):
                # If there's a signature but it's not at the end, clean it up
                parts = generated_text.split("Yours")
                generated_text = parts[0].strip() + "\n\nYours" + parts[1].split("Morrissey")[0] + "Morrissey"
            
            return generated_text
            
        except Exception as e:
            print(f"Error processing generated text: {e}")
            # Fall back to the original text if processing fails
            return generated_text
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling Hugging Face API: {e}")
        # Fallback to a simple template if the API call fails
        return f"""Dear friend,

Your question — "{user_input}" — reminded me of something I once sang:

    "{lyric}"

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

        email = generate_morrissey_style_email(user_input, chunk)
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

# Update the route to use the new generator
@morrissey_api.route("/generate_email", methods=["POST"])
@cross_origin()
def generate_email():
    data = request.get_json()
    user_input = data.get("user_input", "")
    
    # Get the most relevant lyric using your existing similarity search
    user_embedding = model.encode([user_input])[0]
    similarities = cosine_similarity([user_embedding], embeddings)[0]
    most_similar_idx = np.argmax(similarities)
    most_similar_lyric = lyric_chunks[most_similar_idx]
    
    # Generate the email using the API
    email_text = generate_morrissey_style_email(user_input, most_similar_lyric)
    
    return jsonify({
        "email": email_text,
        "lyric": most_similar_lyric,
        "song": line_sources[most_similar_idx]["song"],
        "album": line_sources[most_similar_idx]["album"]
    })