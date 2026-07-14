# DirectoryLoader : 
# DirectoryLoader is a document loader in LangChain used to load multiple documents from a directory (folder).
# It automatically scans the specified directory and loads all matching files.
# It uses the glob parameter to filter files based on their extension or pattern (e.g., *.pdf, *.txt, *.docx).
# The loader_cls parameter specifies which document loader should be used for each file (e.g., PyPDFLoader, TextLoader).
# It converts the loaded files into a list of Document objects.
# DirectoryLoader is commonly used when working with multiple documents in RAG applications.
# It simplifies loading large collections of files without manually loading each one.


from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(len(docs)) 

# output : 1186 
# it means 1186 combined pages from all pdfs 

print(docs[0].page_content) # pehli pdf ke phle page ka content
print(docs[0].metadata) 