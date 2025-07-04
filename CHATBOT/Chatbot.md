
## Gemini AI Chatbot using Python Flask
I have created a personal AI chatbot using Python Flask for the backend and Gemini (Google Generative AI) for the AI responses. The chatbot interface is built using HTML, CSS, and JavaScript, providing a clean and userfriendly web UI.

# Key Features:
```
Frontend:
Built with HTML, CSS and JavaScript
Responsive layout with input field and styled chat window
Users can type messages and receive real time AI replies

Backend:
Developed using Flask (Python)
API endpoint (/chat) processes user messages
Integrates with Gemini API using a valid API key

AI Integration:
Utilizes Google Generative AI (Gemini Pro)
Processes user inputs and returns intelligent, natural language responses
Works like a lightweight ChatGPT clone
```
or  
```
use terminal commands like:
curl -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is AI?"}'
```
## Frontend
I developed a responsive and clean frontend interface for a Personal AI Chatbot using HTML, CSS, and JavaScript. This interface allows users to type and send messages and receive AI responses in a smooth and engaging chat window.

#  Key Components:
 ```
HTML:
Structured layout with a chat container, message display area, and input form.
Simple and semantic markup for easy integration with the backend.

CSS:
Custom styling using internal <style> tags.
Light theme with white background, pale blue buttons, and rounded corners.
Designed for clarity and readability with proper spacing, scrollable message box, and modern feel.

JavaScript:
Handles message sending, receiving responses via fetch() to the Flask API.
Displays user and bot messages dynamically.
Smooth auto-scroll and input field behavior (Enter key support + send button).
```

## Backend
I built the backend of a personal AI chatbot using Python Flask and integrated it with Google’s Gemini API (Generative AI) to generate intelligent and natural-sounding responses.

# Technology Stack:
Flask (Python): Lightweight web framework to handle HTTP requests and render the chatbot interface.

Google Generative AI SDK (google-generativeai): Used to interact with the Gemini Pro language model.

 # Main Backend Features:
 ```
Flask Web Server:

@app.route("/") renders the main HTML chat page.
@app.route("/chat", methods=["POST"]) handles AJAX POST requests from the frontend.

```
Gemini Integration:
Configured with your personal Gemini API key using genai.configure(api_key="YOUR_KEY").
Generates responses to user messages using:

```python
model = genai.GenerativeModel("gemini-pro")
response = model.generate_content(user_message)
```
```
JSON Communication:

Frontend sends a JSON message to /chat
Flask processes it and returns a JSON response with the AI reply

Security Note:
The API key is kept server-side for protection and not exposed to the client/browser.
```

## AI Integration 
In this chatbot project, I integrated Google’s Gemini AI (Generative AI) into the Flask backend to enable real-time, intelligent, and conversational responses.

## Communication flow 
```
User types ➜ Frontend sends request to Flask ➜ Flask passes message to Gemini ➜ Gemini generates a reply ➜ Flask sends reply back ➜ Frontend displays it
```
## Why Gemini?
I built a personal AI chatbot using Flask and connected it to Gemini, Google’s AI model. I chose Gemini because it offers a free API, which is perfect for small projects. It was easy to set up with Python. Gemini’s free tier gave me just enough power to build and test my bot smoothly.





