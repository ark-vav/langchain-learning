# RunnableBranch :
# RunnableBranch is a Runnable that routes the input to one of multiple Runnables based on specified conditions.
# It evaluates the conditions in order and executes the first Runnable whose condition is satisfied.
# If no condition matches, the default Runnable is executed.
# RunnableBranch is used to implement conditional logic and decision-making in LCEL workflows.
# It is useful for creating dynamic workflows where different inputs require different processing paths.
#
# Flow:
#
#              Input
#                │
#         Check Conditions
#         /      |       \
#    Condition1 Condition2 Default
#        │          │         │
#   Runnable A Runnable B Runnable C
#        │          │         │
#        └──────────┴─────────┘
#                │
#             Output

# is wale me ek condition or ek default hai. 


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch


load_dotenv()
 
def word_count(text):
    return len(text.split())

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser) 

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence( report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'})) 


# output : 


# #### Historical Context
#
# - **Handover of Crimea (2014)**: Pro-Russian separatists seized government buildings in Ukraine's Crimean Peninsula, leading to the ousting of pro-Western President Viktor Yanukovych.
# - A referendum on Crimea's reunification with Russia was held, and on March 18, 2014, Russian President Vladimir Putin signed the treaty to annex the peninsula.
#
# - **Donbas Conflict (2014-2022)**: The conflict in the Donbas region, specifically in Luhansk and Donetsk, started after separatists declared independence from Ukraine.
# - Ukraine and its NATO allies accused Russia of direct military involvement, a claim denied by Russia.
#
# #### Key Actors
#
# - **Russian Federation**:
#   - **Government**: Led by President Vladimir Putin, who has supported separatists and been accused of direct military involvement.
#   - **Military**: Allegedly provided significant support to separatist forces, including heavy weapons and special forces.
#   - **Political**: Justified actions through propaganda and diplomatic maneuvers, accusing Ukrainian and Western aggression.
#
# - **Ukraine**:
#   - **Government**: Led by various political figures aiming to strengthen national unity and security.
#   - **Military**: Faced challenges in countering Russian-backed separatists, particularly in eastern regions.
#   - **International Relations**: Sought support from NATO and the European Union, receiving financial and military aid.
#
# - **Western Nations (NATO and EU)**:
#   - **United States and European Union**: Provided political and economic support to Ukraine, including sanctions against Russia and military aid.
#   - **NATO**: Increased its presence in Eastern Europe to bolster Ukraine's security.
#
# The conflict remains a significant challenge to regional stability and international relations.