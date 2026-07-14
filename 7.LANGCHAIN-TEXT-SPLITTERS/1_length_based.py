#langchain_text_splitters 

from langchain_text_splitters import CharacterTextSplitter 

# aur adf koi pdf se mamla ho to ye use krna hai : 

# from langchain_community.document_loaders import PyPDFLoader 
# loader = PyPDFLoader("dl-curriculum.pdf")
# docs = loader.load()

text = """Artificial intelligence has rapidly evolved from a niche field of computer science into a transformative technology
used across industries. Modern AI systems are capable of understanding natural language, recognizing images, generating creative
content, and assisting with complex decision-making. Businesses leverage AI to automate repetitive tasks, improve customer support 
through chatbots, optimize supply chains, and analyze vast amounts of data for actionable insights. As models become more powerful, 
developers are increasingly focusing on building reliable, scalable, and responsible AI applications that can operate in real-world
environments while maintaining transparency, privacy, and security. The emergence of large language models has further accelerated 
innovation by enabling developers to create intelligent assistants, coding copilots, research tools, and autonomous agents capable of 
performing multi-step reasoning and interacting with external systems.

Building production-ready AI applications requires much more than simply connecting an API to a frontend. Developers must design 
efficient data pipelines, implement retrieval systems using vector databases, manage prompts effectively, monitor model performance,
and ensure applications remain reliable under varying workloads. Frameworks such as LangChain and LangGraph simplify the creation of
sophisticated AI workflows by providing abstractions for chaining models, integrating tools, maintaining memory, and orchestrating
multiple agents. As organizations adopt AI at scale, cloud platforms like AWS, Azure, and Google Cloud play a critical role by 
offering managed infrastructure, GPU resources, deployment services, and security features that enable teams to move from 
experimental prototypes to enterprise-grade AI systems."""

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ''
)

result = splitter.split_text(text)
# result = splitter.split_documents(docs)


print(result) 
# print(result[0].page_content)


# The text will be split into multiple chunks of approximately 100 characters each.
# Since chunk_overlap = 0, no characters are repeated between consecutive chunks.
# Since separator = '', the splitter ignores words and sentences and simply cuts the text every ~100 characters.
# Words may be split in the middle because splitting is based purely on character count.
# The result is a Python list of strings, where each string represents one chunk.
# print(result) displays this list of chunks in the terminal.


