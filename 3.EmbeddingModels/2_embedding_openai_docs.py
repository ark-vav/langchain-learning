from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]
result = embedding.embed_documents(documents)

print(str(result))

#example output : 


documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

embeddings = [
    #ouput kuch aisa dikhega ye upr wala sirf understanding ke liye hai 
    # Delhi is the capital of India
    [
        -0.163, 0.276, -0.008, 0.449,
        -0.006, 0.119, -0.027, 0.119,
         0.006, -0.020, 0.148, -0.017,
        -0.145, -0.188, -0.203, -0.182,
         0.119, 0.074, -0.283, 0.029,
        -0.021, 0.072, -0.000, -0.319,
         0.041, 0.321, -0.048, -0.107,
        -0.004, 0.054, 0.131, 0.087
    ],

    # Kolkata is the capital of West Bengal
    [
        -0.158, 0.269, -0.012, 0.441,
        -0.010, 0.114, -0.031, 0.115,
         0.009, -0.023, 0.144, -0.014,
        -0.151, -0.181, -0.196, -0.176,
         0.124, 0.069, -0.277, 0.034,
        -0.018, 0.076, 0.003, -0.311,
         0.045, 0.314, -0.043, -0.101,
        -0.007, 0.059, 0.126, 0.081
    ],

    # Paris is the capital of France
    [
        -0.081, 0.119, 0.064, 0.301,
         0.071, -0.058, 0.121, -0.094,
        -0.033, 0.092, -0.049, 0.183,
         0.081, -0.097, -0.119, -0.042,
         0.224, -0.118, 0.071, -0.051,
         0.119, -0.082, 0.099, -0.133,
         0.156, 0.228, 0.042, -0.073,
         0.061, -0.041, 0.183, -0.016
    ]
]