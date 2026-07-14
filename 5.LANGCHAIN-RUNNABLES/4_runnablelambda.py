# RunnableLambda : 
# RunnableLambda is a Runnable that wraps a Python function so it can be used in an LCEL chain.
# It allows custom Python logic to be integrated into LangChain workflows.
# It takes an input, executes the wrapped function, and returns the function's output.
# RunnableLambda is useful for preprocessing, postprocessing, data transformation, and custom business logic.
# It behaves like any other Runnable and can be chained with prompts, models, parsers, and other Runnables.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda


load_dotenv()
 
def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt | model | parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count) 
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

print(final_chain.invoke({'topic':'AI'}))

# output : 

# {'joke': 'Why did the AI refuse to play hide and seek?\n\nBecause it knew the exact location of every atom in the room!', 'word_count': 22} 