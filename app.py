from flask import Flask
from flask_cors import CORS
from morrisseybot import morrissey_api

app = Flask(__name__)
CORS(app)  # 🚨 Add this line
app.register_blueprint(morrissey_api)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)