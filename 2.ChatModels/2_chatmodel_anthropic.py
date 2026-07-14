from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-3-5-sonnet-20241022')

result = model.invoke("What is the capital of India")

print(result.content)

# input : python 2.ChatModels/2_chatmodel_anthropic.py
# output : New Delhi is the capital of India. 
#  It is located in the northern part of the country and 
#  serves as the seat of the Indian government.
#  New Delhi is part of the larger metropolitan area of Delhi,
#  and it became the capital of India in 1931,
#  replacing Calcutta (now Kolkata) as the capital during British colonial rule.
