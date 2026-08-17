from flask import Flask, render_template, request, jsonify
from rag_engine import RAGEngine
from intent_router import is_claim_query

app = Flask(__name__)

rag = RAGEngine("Chat_Bot_Claims\\DATA\\DATA_FINAL_EXECUTE.pdf")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please enter a valid question."})

    answer = rag.answer(user_message)

    return jsonify({"response": answer})
@app.route("/analysis")
def analysis():
    return jsonify(rag.get_analytics())
if __name__ == "__main__":
    app.run(debug=True)