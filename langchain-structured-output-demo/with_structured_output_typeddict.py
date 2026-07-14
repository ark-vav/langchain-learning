# with_structured_output() makes the LLM return data in a predefined schema instead of plain text,
# making the output reliable and easy for your code to use.


from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv 
from typing import TypedDict

load_dotenv()

model = ChatOpenAI()

# schema 
class Review(TypedDict):

    summary: str
    sentiment: str 

structured_model = model.with_structured_output(Review)

result = model.invoke("""The hardware is great, but the software feels bloated. There are too 
many pre-installed apps that I can't remove. Also, the UI looks outdated 
compared to other brands. Hoping for a software update to fix this.""")

print(result) 

# {'summary': "The hardware is great, but the software feels bloated with too many pre-installed apps that can't be removed.
# The UI looks outdated compared to other brands. Hoping for a software update to fix this.", 'sentiment': 'negative'}

print(result['summary'])

# The hardware is great, but the software feels bloated with too many pre-installed apps that can't be removed.
# The UI looks outdated compared to other brands. Hoping for a software update to fix this. 

print(result['sentiment'])

# negative 


