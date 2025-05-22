from flask import Flask
from flask_cors import CORS
from morrisseybot import morrissey_api
import os

app = Flask(__name__)

# ✅ Responds to all origins for any /api/* route
CORS(app, resources={r"/api/*": {"origins": "https://morrisseybot-ui.vercel.app"}})

app.register_blueprint(morrissey_api)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)