# CSVLoader is a document loader in LangChain used to load data from CSV (Comma-Separated Values) files.
# It reads each row of the CSV file and converts it into a separate Document object.
# Each Document contains:
#   - page_content → The contents of a single row formatted as text.
#   - metadata → Information such as the source file and row number.
# CSVLoader is commonly used to load structured tabular data into LangChain.
# It is useful for RAG applications involving datasets, spreadsheets, customer records, sales data, and other tabular information.
# It automatically preserves the relationship between column names and their corresponding values.

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='6.LANGCHAIN-DOCUMENTS-LOADERS/Social_Network_Ads.csv')

docs = loader.load()

print(len(docs)) #  no. of lines  in csv file 
print(docs[0]) # 1st row dikhega pagecontent or metadata 


# output :

# 400
# page_content='User ID: 15670487
# Gender: Male
# Age: 19
# EstimatedSalary: 85000
# Purchased: 0' metadata={'source': '6.LANGCHAIN-DOCUMENTS-LOADERS/Social_Network_Ads.csv', 'row': 0} 