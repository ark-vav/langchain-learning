# WebBaseLoader : 
# WebBaseLoader is a document loader in LangChain used to load content from web pages (URLs).
# It fetches the HTML content of a webpage and extracts the readable text.
# It converts the extracted webpage content into one or more Document objects.
# Each Document contains:
#   - page_content → The extracted text from the webpage.
#   - metadata → Information such as the source URL and page details.
# WebBaseLoader uses the Requests library to fetch web pages and BeautifulSoup (bs4) to parse HTML.
# It is commonly used to build RAG applications using online documentation, blogs, articles, and websites.
# Some websites may block scraping using CAPTCHAs or bot protection mechanisms.
# Setting a USER_AGENT is recommended to identify your requests when scraping websites.


from langchain_community.document_loaders import WebBaseLoader
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

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://en.wikipedia.org/wiki/Artificial_intelligence'

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser 

text = docs[0].page_content[:3000]

print(chain.invoke({
    "question": "What is this page about?",
    "text": text
}))

#print(chain.invoke({'question':'What is this page about ','text':docs[0].page_content}))


# output : 
# This page is about the Wikipedia article on Artificial Intelligence (AI). The text provides an overview of the article's structure and content,
# which covers various aspects of AI including its goals, techniques, applications, ethics, history, philosophy, and future. 
# It also includes subsections on specific topics such as reasoning and problem-solving, search and optimization, natural language processing, 
# risks and harms, defining AI, machine consciousness, and superintelligence.