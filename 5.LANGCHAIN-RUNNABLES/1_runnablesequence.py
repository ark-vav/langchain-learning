# RunnableSequence : 
# RunnableSequence is a Runnable that executes multiple Runnables sequentially.
# The output of one Runnable becomes the input of the next.
# It is created automatically when using the | operator in LCEL.
# Used for building multi-step workflows where each step depends on the previous step's output.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain.schema.runnable import RunnableSequence 
from langchain_core.runnables import RunnableSequence 


load_dotenv()
 
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=['text']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model , parser )

print(chain.invoke({'topic':'AI'}))

# output :
# This joke plays on the concept of AI (Artificial Intelligence) and its programming, which often aims to be efficient and
# perform tasks based on necessary functions rather than luxury or optional items. The punchline, "Because it heard the seaweed 
# was optional!" is a humorous twist because seaweed is typically not optional at the beach for humans, especially if they're looking 
# to eat or if they're talking about actual seaweed growing in the water. By stating that seaweed is optional, the joke humorously 
# highlights how AI might interpret things differently from humans, leading to a light-hearted amusing scenario.