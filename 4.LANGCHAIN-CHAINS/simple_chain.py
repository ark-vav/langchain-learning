from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Hugging Face Endpoint
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

# Chat Model
model = ChatHuggingFace(llm=llm)

# Prompt
prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

# Output Parser
parser = StrOutputParser()

# LCEL Chain
chain = prompt | model | parser

# Invoke
result = chain.invoke({
    "topic": "simple chain in LangChain"
})

print(result)

# Visualize Chain
chain.get_graph().print_ascii() 

# output : 
# simple_chain.py me rkha hai 