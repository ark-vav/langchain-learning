from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain.schema.runnable import RunnableParallel 
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task= "text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes","quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model | parser, 
    'quiz' : prompt2 | model | parser 
})

merge_chain = prompt3 | model | parser 

chain = parallel_chain | merge_chain

text = """ Artificial Intelligence has rapidly transformed from a niche area of computer science into one of the most influential technologies of the 
21st century. Over the past decade, advancements in machine learning, deep learning, and natural language processing have enabled AI systems to perform 
tasks that were once thought to require uniquely human intelligence, such as understanding language, recognizing images, generating creative content,
 and making complex decisions. Businesses across industries are increasingly adopting AI to automate repetitive tasks, improve customer experiences,
   optimize supply chains, and uncover valuable insights from massive amounts of data. In healthcare, AI assists doctors by analyzing medical images,
     predicting diseases, and recommending personalized treatments. In finance, it helps detect fraudulent transactions, assess credit risks, and automate 
     trading strategies. Education has also benefited from AI-powered tutoring systems that adapt to individual learning styles, while transportation is
       being reshaped by autonomous driving technologies and intelligent traffic management systems. Despite these remarkable achievements,
         AI also presents significant challenges. Concerns surrounding privacy, algorithmic bias, misinformation, job displacement,
           and ethical decision-making continue to spark debates among researchers, governments, and the public. Responsible AI development requires 
           transparency, fairness, accountability, and strong regulatory frameworks to ensure that these technologies benefit society as a whole.
             Furthermore, the emergence of agentic AI systems capable of planning, reasoning, and using external tools has opened a new frontier in
               artificial intelligence. Frameworks such as LangChain and LangGraph allow developers to build sophisticated AI applications that combine 
               language models with memory, tools, retrieval systems, and complex workflows. These agentic systems can perform multi-step reasoning,
                 collaborate with other agents, and execute tasks autonomously while maintaining context throughout long conversations. As AI models
                   continue to become more powerful and efficient, the demand for engineers skilled in prompt engineering, retrieval-augmented generation 
                   (RAG), vector databases, AI agents, cloud deployment, and production monitoring is expected to grow rapidly. While AI is unlikely to
                     replace every human job, it will undoubtedly reshape the workforce by augmenting human capabilities and creating entirely new career 
                     opportunities. Individuals who continuously learn and adapt to these technological changes will be better positioned to thrive in the
                       evolving digital economy.
"""

result = chain.invoke({'text' : text})

print(result) 

chain.get_graph().print_ascii() 

# output :
# parallel_chain.txt me hai 