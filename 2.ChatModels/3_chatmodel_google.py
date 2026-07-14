from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

result = model.invoke("What is the capital of India")

print(result.content)

#input : python 2.ChatModels/3_chatmodel_google.py 
#output : The capital of India is New Delhi.


 
