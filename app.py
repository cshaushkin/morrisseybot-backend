from flask import Flask, request, jsonify
from flask_cors import CORS
from morrisseybot import morrissey_api
import os

app = Flask(__name__)

# ✅ Allow Vercel frontend to make requests
CORS(app, origins=["https://morrisseybot-ui.vercel.app"], supports_credentials=True)

# ✅ Register the API blueprint
app.register_blueprint(morrissey_api)

# ✅ Preflight CORS for OPTIONS requests
@app.before_request
def handle_options_request():
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS preflight accepted"})
        response.headers.add("Access-Control-Allow-Origin", "https://morrisseybot-ui.vercel.app")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response

# ✅ Dynamic port binding for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)