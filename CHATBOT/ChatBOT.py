from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)

# Configure Gemini
genai.configure(api_key="AIzaSyBsHopsz38ewyZB068aDaIiZIqJhfEHOcY")
model = genai.GenerativeModel("gemini-2.5-flash")
chat = model.start_chat()

@app.route('/')
def home():
    return render_template("chatbot.html")

@app.route('/chat', methods=['POST'])
def chat_gemini():
    data = request.get_json()
    print("User message:", data)
    try:
        user_message = data.get("message", "")
        response = chat.send_message(user_message)
        cleaned = response.text.replace("**", "").replace("*", "").strip()
        return jsonify({"reply": cleaned})
    except Exception as e:
        print("Error:", e)
        return jsonify({"reply": " Server error: " + str(e)})

if __name__ == '__main__':
    app.run(debug=True)
