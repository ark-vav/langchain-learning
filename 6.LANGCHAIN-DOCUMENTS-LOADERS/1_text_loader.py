# TextLoader : 
# TextLoader is a document loader used to load plain text (.txt) files into LangChain.
# It reads the contents of a text file and converts them into one or more Document objects.
# Each Document contains the text as page_content and associated metadata such as the file source.
# TextLoader is commonly used as the first step in RAG pipelines for loading textual data.
# It supports different file encodings and can lazily load documents when required.
#
# Flow:
#
# Text File (.txt)
#        │
#    TextLoader
#        │
#  Document Object(s)
#        │
#  Text Splitter / Embeddings / LLM


from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader(
    "6.LANGCHAIN-DOCUMENTS-LOADERS/cricket.txt"
)

docs = loader.load()

#print(docs)

# output : cricket.txt as a document load hojayega in the form of list
# in short a list of document will be formed , humare is case me bs ek hi document hai list me i.e cricket.txt wali poem 
# document ko load krne pe uska page content and metadata aata hai 

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content})) 

# output : 

# **Summary of "The Gentleman's Symphony"**
#
# The poem "The Gentleman's Symphony" is a lyrical and evocative depiction of cricket, intertwining the game's physical and emotional aspects.
# The sun rises over a cricket field, awakening the spirit of the game through natural imagery and the beauty of human interaction.
# The poem weaves together the perspectives of players, ball, seam, and fielders, creating a symphony of movement and emotion.
#
# Key elements include:
#
# - **The Field and Players:** The field is described as a stage where countless hearts strive and refuse to yield, with players like the bowler, batter, and keeper showcasing skill and determination.
#
# - **The Game's Mechanics:** The ball, seam, and fielding are personified, with the seam spinning tales and the fielders reading the batter's mind.
# - The game's intensity and drama are vividly portrayed through detailed descriptions of shots and catches.
#
# - **The Stadium and Spectators:** The stadium comes alive with the crowd's reactions, from eruptions of joy to the roar of approval.
# - The atmosphere is one of shared experience and camaraderie, transcending the game's outcomes.
#
# - **Youth and Legacy:** The poem highlights the game's impact on young players, from those playing in dusty streets to a child mimicking the stars, capturing the timeless nature and universal appeal of cricket.
#
# - **Game Formats and Spirit:** The poem touches on the different formats of the game (Test matches, One Day, T20) and the enduring spirit that remains constant, emphasizing respect, perseverance, and the personal growth fostered by the game.
#
# - **Reflections and Lessons:** Cricket is portrayed as a metaphor for life, teaching lessons of patience, courage, and humility.
# - It is more than just a game; it is a story written within, a fire that every heart can claim.
#
# Ultimately, the poem celebrates the beauty, drama, and enduring spirit of cricket, suggesting that the game's essence transcends time and format.