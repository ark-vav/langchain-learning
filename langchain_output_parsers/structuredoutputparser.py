from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
#from langchain_core.prompts import PromptTemplate
#from langchain_core.output_parsers import JsonOutputParser


load_dotenv()
 
# Define the model 
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema 

#nai chalega out dated code hai
