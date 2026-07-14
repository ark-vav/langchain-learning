from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
 
# Define the model 
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables={}, 
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser 

result = chain.invoke({})

print(result) 

# output : 
# {'name': 'Liam Chen', 'age': 32, 'city': 'Shanghai'}

# result type = <class 'dict'>



