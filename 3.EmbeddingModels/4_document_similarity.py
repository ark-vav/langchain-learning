#1. Initialize the embedding model.
#2. Convert all documents into embeddings and store them.
#3. Convert the user's query into an embedding using the same model.
#4. Compute the cosine similarity between the query embedding and every document embedding.
#5. Rank the documents based on similarity scores.
#6. Retrieve and display the document with the highest similarity.


from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about virat kohli'

#finding documents embedding = 5 300 dimensional space vectors 
doc_embeddings = embedding.embed_documents(documents)

#finding query embedding 
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])


#input query : 'tell me about virat kohli' 
#output : 
# tell me about virat kohli 
# Virat Kohli is an India cricketer known for his aggressive batting and leadership.
# similarity score is: 0.6631398845093458 

#input query : 'tell me about bumrah'
#output :
# tell me about bumrah 
# Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers.
# similarity score is: 0.684336345376626

