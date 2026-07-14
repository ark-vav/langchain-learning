#Runnablepassthrough : 
# RunnablePassthrough is a Runnable that passes the input unchanged to the next Runnable.
# It does not modify or process the input.
# It is used to preserve the original input while combining it with outputs from other Runnables.
# RunnablePassthrough is commonly used in RAG pipelines and complex LCEL workflows.
# It is useful when the original input is needed later in the chain.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain.schema.runnable import RunnableSequence 
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough


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
    template='Explain the following joke - {text}',
    input_variables=['text']
)

joke_gen_chain = RunnableSequence(prompt1, model , parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model ,parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'cricket'})) 


# output :
{
# 'joke': 'Why did the cricket refuse to play baseball?

# Because he didn\'t want to be called "the bug" in the outfield!',

# 'explanation': 'This joke is a play on words and stereotypes.

# In baseball, the "outfield" refers to the area of the field beyond the grass line where the outfielders (usually the right fielder, center fielder, and left fielder) play.

# The word "outfield" is also a pun because it sounds like "out-fied," which could be interpreted as getting gotten rid of or being excluded.

# The joke is that crickets are often referred to as "bugs," and the cricket in the joke is a bit sensitive about it.

# He doesn\'t want to be called "the bug" in the outfield because it would be a reminder of his insect nature, which he might not want to be associated with while playing baseball.'
} 