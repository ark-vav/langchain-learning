from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

# dimensions parameter is used to explicitly control the size of the embedding vector returned by OpenAI's text-embedding-3-large model 
# By default, text-embedding-3-large generates high-resolution embeddings with 3,072 dimensions. 
# By passing dimensions=32,you are compressing that vector down to just 32 numbers.

result = embedding.embed_query("Delhi is the capital of India")

print(str(result))

#32 dimensional vector representing the contextual meaning of the query with be obtained 

# input : python 3.EmbeddingModels/1_embedding_openai_query.py 
# output : 
# [-0.16284997761249542, 0.2760346233844757, -0.007678520400077105, 0.44934046268463135,
#  -0.005889588501304388, 0.11854328215122223, -0.02743028849363327, 0.1193923791792429,
#   0.0064205499365988, 0.11854328215122223, -0.02743028849363327, 0.1193923791792429, 
#   0.006420549936592579, -0.02032357268035412, 0.14847317337989807, -0.016811057925224304,
#  -0.1453364193494836, -0.18781334161758423, -0.20349712669849396, -0.18219330908694396,
#   0.119392819797674103, 0.074301935723263465, -0.28333259373474, 0.029211051762104034,
#  -0.02050328254699707, 0.07214541733264923, -0.00011640309821814299, -0.31916505098342896,
#   0.041091519594192505, 0.32073344280014038, -0.0483539816333186, -0.1066494130984184,
#  -0.00373975158754, 0.05446848273277283, 0.130959615111351]
