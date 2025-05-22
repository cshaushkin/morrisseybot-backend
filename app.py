from flask import Flask
from flask_cors import CORS
from morrisseybot import morrissey_api

app = Flask(__name__)

# ✅ Enable CORS with full preflight support for Vercel
CORS(app, resources={r"/api/*": {"origins": "https://morrisseybot-ui.vercel.app"}}, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

app.register_blueprint(morrissey_api)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)