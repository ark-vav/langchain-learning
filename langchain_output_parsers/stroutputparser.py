from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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
    template="Write a 4 line summary on the following text. /n {text}",
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result1 = model.invoke(prompt2)

print(result1.content)

# output : 
# Black holes are incredibly dense regions in space with gravity so strong that nothing can escape, including light. 
# They form under specific astrophysical conditions, such as the collapse of massive stars or from mergers in the early universe. 
# Key properties include the event horizon, where the escape velocity equals the speed of light, and the singularity at the center, where the density is infinite. 
# Black holes also exhibit thermodynamic properties, emitting radiation known as Hawking radiation.

