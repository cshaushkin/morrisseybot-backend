from flask import Flask
from flask_cors import CORS
from morrisseybot import morrissey_api

app = Flask(__name__)

# ✅ Correctly configure CORS to handle preflight
CORS(app, resources={r"/api/*": {"origins": "https://morrisseybot-ui.vercel.app"}}, supports_credentials=True)

app.register_blueprint(morrissey_api)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)