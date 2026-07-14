from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv

load_dotenv()

# Qwen 2.5 is fully supported on the new serverless router architecture
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    #we were using tinyllama first 
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")
print(result.content)