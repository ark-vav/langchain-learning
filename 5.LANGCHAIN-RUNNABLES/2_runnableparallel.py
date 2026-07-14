# RunnableParallel : 
# RunnableParallel is a Runnable that executes multiple Runnables simultaneously.
# It sends the same input to all Runnables in parallel.
# Each Runnable processes the input independently.
# The outputs are collected and returned as a dictionary.
# It is used when multiple tasks can be performed independently on the same input.
# RunnableParallel is useful for improving efficiency by running independent operations concurrently.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain.schema.runnable import RunnableSequence 
from langchain_core.runnables import RunnableSequence, RunnableParallel


load_dotenv()
 
prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a linkedin post about {topic}',
    input_variables=['topic']
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic':'AI'})

print(result)

# output : 

{
# 'tweet': "🚀 Exciting times ahead as AI continues to revolutionize the way we live and work! From healthcare to entertainment,
# the possibilities are limitless. Let's embrace the future together! #ArtificialIntelligence #Innovation 🌟💡",

# 'linkedin': "🚀 Revolutionizing the Future: The Impact of AI on Our World 🚀

# In today's rapidly evolving digital landscape, Artificial Intelligence (AI) is no longer just a buzzword—it's a transformative force that's reshaping industries, enhancing productivity, and opening new doors for innovation. As we stand on the brink of an AI-driven future, it's crucial to understand how this technology is impacting our professional and personal lives.

# ✨ Key Areas Where AI is Making a Difference:

# 1. **Customer Service**: Chatbots and virtual assistants are providing 24/7 support, improving customer satisfaction.

# 2. **Healthcare**: AI is aiding in disease diagnosis, personalized treatment plans, and even drug discovery.

# 3. **Finance**: Fraud detection, risk assessment, and automated trading are just some of the ways AI is streamlining financial operations.

# 4. **Transportation**: Autonomous vehicles are revolutionizing how we move people and goods, potentially reshaping urban landscapes.

# 5. **Manufacturing**: AI-driven automation and predictive maintenance are boosting efficiency and reducing downtime.

# 💡 As professionals, we must stay informed and adapt to these changes. Whether you're a tech enthusiast, a business leader, or simply curious about the future, the role of AI in society is undeniable.

# 🌟 Let's continue to explore and harness the power of AI for the betterment of our world. What are your thoughts on the future of AI? Share your insights and experiences below!

# #AI #Technology #Innovation #FutureOfWork #CareerDevelopment

# [LinkedIn Profile Picture of Yourself or Company Logo]"
}