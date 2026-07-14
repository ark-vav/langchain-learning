# PyPDFLoader : 
# PyPDFLoader is a document loader in LangChain used to load content from PDF files.
# It extracts text from a PDF and converts each page into a separate Document object.
# Each Document contains:
#   - page_content → The extracted text from that page.
#   - metadata → Information such as page number and source file.
# PyPDFLoader is commonly used as the first step in RAG pipelines for loading PDF documents.
# It internally uses the PyPDF library for text extraction.
#
# Example Output:
#
# [
#   Document(
#       page_content="Text from page 1",
#       metadata={"page": 0, "source": "file.pdf"}
#   ),
#
#   Document(
#       page_content="Text from page 2",
#       metadata={"page": 1, "source": "file.pdf"}
#   ),
#
#   ...
# ]
#
# Limitations:
# - Uses the PyPDF library under the hood.
# - Not suitable for scanned PDFs or image-based PDFs.
# - May not accurately extract text from complex layouts, tables, or multi-column documents.
#
# Flow:
#
# PDF File (.pdf)
#        │
#    PyPDFLoader
#        │
# Document Object(s)
# (One per page)
#        │
# Text Splitter / Embeddings / LLM


from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    "6.LANGCHAIN-DOCUMENTS-LOADERS/PYTHON_PROGRAMMING_NOTES.pdf"
)

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)

# output : 

# 142
#
# PYTHON PROGRAMMING                                      III YEAR/II SEM                  MRCET
#
# PYTHON PROGRAMMING
#
# [R17A0554]
#
# LECTURE NOTES
#
# B.TECH III YEAR – II SEM (R17)
#
# (2019-20)
#
# DEPARTMENT OF
#
# COMPUTER SCIENCE AND ENGINEERING
#
# MALLA REDDY COLLEGE OF ENGINEERING & TECHNOLOGY
#
# (Autonomous Institution – UGC, Govt. of India)
#
# Recognized under 2(f) and 12 (B) of UGC ACT 1956
#
# (Affiliated to JNTUH, Hyderabad, Approved by AICTE - Accredited by NBA & NAAC – ‘A’ Grade - ISO 9001:2015 Certified)
#
# Maisammaguda, Dhulapally (Post Via. Hakimpet), Secunderabad – 500100, Telangana State, India
#
# {
#     'producer': 'https://www.convertapi.com',
#     'creator': '',
#     'creationdate': '2019-12-09T08:16:17+02:00',
#     'author': 'Windows User',
#     'moddate': '2019-12-09T08:16:19+02:00',
#     'source': '6.LANGCHAIN-DOCUMENTS-LOADERS/PYTHON_PROGRAMMING_NOTES.pdf',
#     'total_pages': 142,
#     'page': 1,
#     'page_label': '2'
# }



# | Use Case                         | Recommended Loader                              |
# |----------------------------------|-------------------------------------------------|
# | Simple, clean PDFs               | PyPDFLoader                                     |
# | PDFs with tables/columns         | PDFPlumberLoader                                |
# | Scanned/image PDFs               | UnstructuredPDFLoader or AmazonTextractPDFLoader |
# | Need layout and image data       | PyMuPDFLoader                                   |
# | Want best structure extraction   | UnstructuredPDFLoader                           |