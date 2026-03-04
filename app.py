import os
import uuid
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from rag_engine import RAGEngine
from intent_router import is_claim_query
load_dotenv()
app = Flask(__name__)
# Initialize RAG Engine
rag = RAGEngine("DATA\\Claimsss.pdf")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "").strip()

        if not user_message:
            return jsonify({"response": "Please enter a valid question."})

        # Direct answer (no session memory)
        if is_claim_query(user_message):
            answer = rag.answer(user_message)
        else:
            answer = rag.answer(user_message)

        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

@app.route("/analysis")
def analysis():
    return jsonify(rag.get_analytics())

if __name__ == "__main__":
    app.run(debug=True)