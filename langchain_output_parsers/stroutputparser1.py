from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
 
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report 
template1= PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

# 2nd prompt -> summary 
template2= PromptTemplate(
    template="Write a 3 line summary on the following text.\n{text}",  
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser 

result = chain.invoke({'topic':'black hole '})

print(result) 


# output: 
# Black holes, first theorized in the 18th century and confirmed in the 20th, are regions with immense gravitational fields from which nothing can escape. 
# They come in various types, including stellar-mass (5-100 solar masses), supermassive (millions to billions of solar masses), and intermediate or primordial. 
# Key properties such as the event horizon and singularity define their unique characteristics, posing challenges for theoretical physics.

# Use StrOutputParser when you want plain text from the model without metadata. 
# If you want structured output instead (like JSON or a Pydantic model), 
# you'd use a different parser such as JsonOutputParser or structured output support with a Pydantic model. 

