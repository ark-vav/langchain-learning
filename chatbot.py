#chatbot 

from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load API Key
load_dotenv()

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Store conversation history
chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        break

    # Add user's message
    chat_history.append(HumanMessage(content=user_input))

    # Get model response using entire conversation
    result = model.invoke(chat_history)

    # Store AI response
    chat_history.append(AIMessage(content=result.content))

    print("AI:", result.content)

print("\nChat History:\n")

for message in chat_history:
    if isinstance(message, HumanMessage):
        print("You:", message.content)
    else:
        print("AI :", message.content) 





# output without context memory : 
# Type 'exit' to quit. 
# You: what is the capital of india ?
# AI : The capital of India is New Delhi.
# You: how many states in india ?
# AI : India consists of 28 states and 8 union territories. The states are divided into various administrative regions based on geographical, cultural, and historical factors. The union territories are governed by the union government but have their own local administrations.
# You: is 9 a prime number 
# AI : No, 9 is not a prime number. A prime number is defined as a number greater than 1 that has no positive divisors other than 1 and itself. However, 9 can be divided evenly by 1, 3, and 9. Since it has divisors other than 1 and itself, it is classified as a composite number.
# You: exit 

# output with context memory : 

# Type 'exit' to quit.

# You: hi
# AI: Hello! How can I assist you today?
# You: which one is greater 2 or 0 
# AI: The number 2 is greater than the number 0.
# You: now multiply the bigger number with 10 
# AI: The bigger number is 2, and when you multiply it by 10, you get:

# \[ 2 \times 10 = 20 \]
# You: exit 

# Chat History:

# You: hi
# AI : Hello! How can I assist you today?
# You: which one is greater 2 or 0
# AI : The number 2 is greater than the number 0.
# You: now multiply the bigger number with 10
# AI : The bigger number is 2, and when you multiply it by 10, you get:

# \[ 2 \times 10 = 20 \]