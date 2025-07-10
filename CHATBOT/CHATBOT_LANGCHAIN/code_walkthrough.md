
#  LangChain Chatbot using Gemini API – Walkthrough


- [LangChain + Gemini Chatbot Walkthrough](https://github.com/Shishirise/ML-with-Python/blob/main/Frameworks/Langchain/Langchain.md)

---

##  Requirements

Install the required Python packages:

```bash
pip install langchain google-generativeai langchain-google-genai
```

## Code Walkthrough

#Import necessary components from LangChain and the Google Generative AI wrapper
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
ChatGoogleGenerativeAI: LangChain wrapper for Gemini LLMs

ConversationBufferMemory: Stores full conversation history

ConversationChain: Connects LLM + memory into a functional chatbot
```

```python

# Your Google API Key 
API_KEY = "API_KEY"
```



# Initialize the LLM using the Gemini 2.5 Flash model
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",        # Fast Gemini model
    google_api_key=API_KEY          # Auth using your API key
)
```

# Create memory to store conversation history (full message history)
```python
memory = ConversationBufferMemory()
```
Keeps track of the entire conversation.

Useful for context-aware responses.


# Create the conversation chain that combines the LLM and memory
```python
chain = ConversationChain(
    llm=llm,                         # Use Gemini model
    memory=memory                   # Store chat history
)
```


# Start the chatbot interaction loop
```python
print("Chatbot")
```
Prints the chatbot intro.

```python
while True:
    # Get input from the user
    user_input = input("You: ")

    # Exit the loop if the user types 'exit'
    if user_input.lower() in ["exit"]:
        print("Chat ended.")
        break

    # Send the user's message to the conversation chain
    # Input must be in dictionary form with key 'input'
    response = chain.invoke({"input": user_input})

    # Print the AI's response
    print("AI:", response["response"])
Runs continuously until you type exit.
```
