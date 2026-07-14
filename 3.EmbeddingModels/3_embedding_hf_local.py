from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]
vector = embedding.embed_documents(documents)

#text = "Delhi is the capital of India"
#vector= embedding.embed_query(text)

print(str(vector))



# input : python 3.EmbeddingModels/3_embedding_hf_local.py 
# embedding.embed_query(text) wale me output : 384 dimensional vector string aa jayegi aisi [0.04354959726333618, 0.023877238854765892,...... 0.047101233154535294, 0.04865921288728714]
# embedding.embed_documents(documents) wale me output : 384 dimensional * 3 wali strings aa jayegi [[384],[384],[384]]
# and ye sb 2d list hai 