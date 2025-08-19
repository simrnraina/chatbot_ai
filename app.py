from flask import Flask, request, jsonify, send_from_directory
from twilio.twiml.messaging_response import MessagingResponse
import os
from chatbot import get_response   # Import your chatbot response function

app = Flask(__name__, static_folder="static")

# ------------------------------
# Serve Frontend
# ------------------------------
@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

# ------------------------------
# API for Frontend Chat
# ------------------------------
@app.route("/get_response", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message")
    bot_reply = get_response(user_message)
    return jsonify({"reply": bot_reply})

# ------------------------------
# Twilio Webhook (SMS/WhatsApp)
# ------------------------------
@app.route("/twilio-webhook", methods=["POST"])
def twilio_webhook():
    incoming_msg = request.form.get("Body")
    bot_reply = get_response(incoming_msg)

    resp = MessagingResponse()
    resp.message(bot_reply)
    return str(resp)

# ------------------------------
# Run Locally
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
