## Top-Level Imports
```python

from flask import Flask, request, jsonify, render_template
```
We're importing Flask and some of its functions:

Flask – to create the web app.

request – to receive data from the client (like messages).

jsonify – to send JSON responses back.

render_template – to load your HTML file (chatbot.html).

```python
import google.generativeai as genai
```
This imports Google’s Gemini API (Generative AI tools). You'll use it to generate responses.


```python
from flask_cors import CORS
```
CORS (Cross-Origin Resource Sharing) is used so your frontend (HTML/JS) can talk to your backend Flask app without security issues, especially if they are on different domains/ports.

# App Setup
```python

app = Flask(__name__, template_folder='templates')
```
This creates your Flask app and tells it where to find your HTML files (in a folder called templates).

CORS(app)
Enables CORS support for your app so it doesn’t block incoming requests from other domains (like your local JS code calling the Python backend).

# Gemini Configuration
```python
genai.configure(api_key="API_KEYS")
```
This sets up the connection to Google’s Gemini API using your API key. Replace "API_KEYS" with your actual key.

```python
model = genai.GenerativeModel("gemini-2.5-flash")
```
Loads a specific Gemini model called "gemini-2.5-flash" which is fast and optimized for real-time interaction.

```python
chat = model.start_chat()
```
Starts a new conversation session with the Gemini model. This keeps context across multiple messages (just like a real chat).

# Routes
```python
@app.route('/')
def home():
    return render_template("chatbot.html")
```
This defines the homepage of your app.
When someone goes to / in the browser, it will show the chatbot HTML page (frontend).

```python

@app.route('/chat', methods=['POST'])
def chat_gemini():
```
This is the route that handles chatting.
It listens for POST requests sent to /chat, usually from JavaScript (AJAX/fetch) in your HTML.

```python

    data = request.get_json()
    print("User message:", data)
```
Reads the user’s message (in JSON format).
Prints it to your terminal or console so you can debug or monitor activity.

```python
 try:
user_message = data.get("message", "")
```
Gets the actual text message from the received JSON. If none is sent, it defaults to an empty string.

```python
response = chat.send_message(user_message)
```
Sends the user’s message to the Gemini chatbot model and gets a response back.

```python

cleaned = response.text.replace("**", "").replace("*", "").strip()
```
Cleans the response text by removing any Markdown formatting like **bold** or *italic*.

```python
return jsonify({"reply": cleaned})
```
Sends the cleaned response back to the frontend in JSON format (as { reply: "response from Gemini" }).

```python

except Exception as e:
print("Error:", e)
return jsonify({"reply": " Server error: " + str(e)})
```
If anything goes wrong (like a network error or bad input), it will catch the error and return a user-friendly message with the error details.

# Running the App
```python

if __name__ == '__main__':
    app.run(debug=True)
```

This starts the Flask app only if you're running the script directly (not importing it elsewhere).
debug=True allows you to see helpful error messages and auto-restarts the server when you make code changes.


