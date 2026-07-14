# DOCUMENT STRUCTURE BASED TEXT SPLITTER

# Document Structure Based Text Splitter splits documents according to their natural structure instead of fixed character lengths.

# It preserves logical sections such as headings, paragraphs, chapters, or markdown sections.

# This approach keeps related information together, improving context for LLMs.

# It is commonly used for Markdown, HTML, JSON, XML, PDFs, and structured documents.

# Unlike CharacterTextSplitter, it understands document hierarchy.

# Main Goal:
# Keep semantically related content together by using the document's existing structure.

# Advantages:
# Better context preservation.
# Higher retrieval accuracy in RAG.
# Less information loss.
# More meaningful chunks.

# Disadvantages:
# Chunk sizes may become uneven.
# Depends on proper document formatting.

# Common Structure-Based Splitters in LangChain:

# MarkdownHeaderTextSplitter
# Splits Markdown files using headings (#, ##, ###).

# HTMLHeaderTextSplitter
# Splits HTML documents using HTML tags like <h1>, <h2>, <h3>.

# RecursiveJsonSplitter
# Splits nested JSON objects while preserving hierarchy.

# HTMLSectionSplitter
# Splits HTML into logical sections.

# These splitters attach metadata such as section names or headers to every chunk.

# Example:

# Input Markdown

# Introduction
# AI is transforming industries.

# Machine Learning
# ML enables computers to learn from data.

# Deep Learning
# Uses neural networks.

# Output Chunks

# Chunk 1
# Header: Introduction
# AI is transforming industries.

# Chunk 2
# Header: Machine Learning
# ML enables computers to learn from data.

# Chunk 3
# Header: Deep Learning
# Uses neural networks.

# Metadata Example

# {
#   "Header 1": "Machine Learning"
# }

# Benefits in RAG

# Easier semantic search.
# Better retrieval quality.
# LLM understands which section information came from.
# Prevents unrelated topics from mixing together.

# Best Use Cases

# Documentation
# Books
# Research papers
# Markdown notes
# HTML webpages
# Technical manuals
# Wikis
# API documentation

# Comparison

# CharacterTextSplitter
# Splits by character count.

# RecursiveCharacterTextSplitter
# Tries multiple separators while respecting chunk size.

# TokenTextSplitter
# Splits by token count.

# Document Structure Splitter
# Splits using document hierarchy like headings and sections.

# Summary

# Uses document structure instead of character limits.
# Produces semantically meaningful chunks.
# Preserves headings and metadata.
# Improves retrieval performance in RAG systems.
# Best for structured documents like Markdown, HTML, and JSON.

from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_document = """
# LangChain

LangChain is a framework for building LLM applications.

## Text Splitters

Text splitters divide long documents into smaller chunks.

### CharacterTextSplitter

Splits text based on characters.

### RecursiveCharacterTextSplitter

Splits text using multiple separators.

## Vector Stores

Vector stores store embeddings for semantic search.
"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

docs = splitter.split_text(markdown_document)

for i, doc in enumerate(docs, start=1):
    print(f"\nChunk {i}")
    print("-" * 40)
    print(doc.page_content)
    print("Metadata:", doc.metadata)


# Chunk 1
# ----------------------------------------
# LangChain is a framework for building LLM applications.
# Metadata: {'Header 1': 'LangChain'}

# Chunk 2
# ----------------------------------------
# Text splitters divide long documents into smaller chunks.
# Metadata: {'Header 1': 'LangChain', 'Header 2': 'Text Splitters'}

# Chunk 3
# ----------------------------------------
# Splits text based on characters.
# Metadata: {'Header 1': 'LangChain', 'Header 2': 'Text Splitters', 'Header 3': 'CharacterTextSplitter'}

# Chunk 4
# ----------------------------------------
# Splits text using multiple separators.
# Metadata: {'Header 1': 'LangChain', 'Header 2': 'Text Splitters', 'Header 3': 'RecursiveCharacterTextSplitter'}

# Chunk 5
# ----------------------------------------
# Vector stores store embeddings for semantic search.
# Metadata: {'Header 1': 'LangChain', 'Header 2': 'Vector Stores'}


from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name
    
    def is_passing(self):
        return self.grade >= 6.0 
    
# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
"""
# Initialize the splitter 
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

# Perform the split 
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[1])


