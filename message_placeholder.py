# MessagesPlaceholder is used to dynamically insert a variable list of past chat messages into a prompt template.

# Reserving a Spot for History: Inside ChatPromptTemplate, it sets up a placeholder named 'chat_history'. 
# This tells LangChain: "An unspecified number of previous messages will be injected right here between the 
# system prompt and the user's current query."

# Handling Raw Message Objects: Instead of forcing you to format past conversations into a single string,
# it accepts an array containing structured LangChain message objects (like HumanMessage and AIMessage) and renders them seamlessly.

# Enabling Chat Continuity: As seen in the terminal output, when chat_history (loaded from chat_history.txt) is passed into the template,
# MessagesPlaceholder expands it to include the past support request and the AI's response, allowing the model 
# to understand the context of the new query ("Where is my refund?").


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage 
# chat template 
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agents'),
    MessagesPlaceholder(variable_name='chat_history'), 
    ('human','{query}')
])

chat_history = []
# load chat history 
with open('chat_history.txt') as f :
    chat_history.extend(f.readlines())

print(chat_history)

# create prompt 
prompt = chat_template.invoke({'chat_history':chat_history,'query':'Where is my refund '})

print(prompt) 