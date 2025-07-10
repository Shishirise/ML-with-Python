# Import necessary components from LangChain and the Google Generative AI wrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Your Google API Key 
API_KEY = "API_KEY 

# Initialize the LLM using the Gemini 2.5 Flash model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",        # Use a fast and lightweight Gemini model
    google_api_key=API_KEY          # API key for authentication
)

# Create memory to store conversation history (full message history)
memory = ConversationBufferMemory()

# Create the conversation chain that combines the LLM and memory
chain = ConversationChain(
    llm=llm,                         # The Gemini LLM to use
    memory=memory                   # Memory to keep track of past messages
)

# Start the chatbot interaction loop
print("Chatbot")

while True:
    # Get input from the user
    user_input = input("You: ")
    
    # Exit the loop if the user types 'exit'
    if user_input.lower() in ["exit"]:
        print("Chat ended.")
        break
    
    # Send the user's message to the conversation chain
    # The input must be in dictionary form with key 'input'
    response = chain.invoke({"input": user_input})
    
    # Print the AI's response
    print("AI:", response["response"])