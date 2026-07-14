from langchain_openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct') 

result = llm.invoke("What is the capital of India") 

print(result) 

#the result will be like " The capital of India is New Delhi. "
#as we know llms are outdated so we will use chatmodels instead 
#and as we can see that llm takes simple input string and gives simple output string 

