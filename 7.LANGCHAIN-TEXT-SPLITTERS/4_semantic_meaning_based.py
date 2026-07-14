# SEMANTIC MEANING BASED TEXT SPLITTER

# Semantic Meaning Based Text Splitter divides text according to its meaning rather than character count or document structure.

# It uses embedding models to understand the semantic similarity between sentences.

# Sentences discussing the same topic are grouped into the same chunk.

# When the meaning changes significantly, a new chunk is created.

# LangChain provides SemanticChunker for semantic-based splitting.

# SemanticChunker is available in the langchain-experimental package.

# It requires an embedding model such as OpenAIEmbeddings or HuggingFaceEmbeddings. 


from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

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
    print("The student is not passing.")
"""

embeddings = OpenAIEmbeddings()

splitter = SemanticChunker(embeddings)

chunks = splitter.create_documents([text])

print(len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}")
    print("-" * 40)
    print(chunk.page_content)


# output : 
# 2
#
# Chunk 1
# ----------------------------------------
# class Student:
#     def __init__(self, name, age, grade):
#         self.name = name
#         self.age = age
#         self.grade = grade
#
#     def get_details(self):
#         return self.name
#
#     def is_passing(self):
#         return self.grade >= 6.0
#
# Chunk 2
# ----------------------------------------
# # Example usage
# student1 = Student("Aarav", 20, 8.2)
# print(student1.get_details())
#
# if student1.is_passing():
#     print("The student is passing.")
# else:
#     print("The student is not passing.")