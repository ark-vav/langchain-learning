from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv

load_dotenv()

model= ChatOpenAI(model='gpt-4')

result = model.invoke("What is the capital of India?")

print(result)

#input will be like this : PS C:\Users\vaibh\Desktop\langchain_models> python 2.ChatModels/1_chatmodel_openai.py

#result will be like this : 
#content='The capital of India is New Delhi.' additional_kwargs={'refusal': None} response_metadata={'token
#usage': {'completion_tokens': 9, 'prompt_tokens': 14, 'total_tokens': 23, 'completion_tokens_details': {'
#accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}
#, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0613', 'system_f
#ingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-50960ad6-1055-4a0a-8a70-71e0b04ee4b4
#-0' usage_metadata={'input_tokens': 14, 'output_tokens': 9, 'total_tokens': 23, 'input_token_details': {'a
#udio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}

#as we can see that result of chatmodel is not just simple string like llm, instead we have alot of metadata
# and we were to only see the final result string then we will write 
print(result.content) 
# then result will be just : The capital of India is New Delhi. 


#The temperature is a parameter that controls the randomness of a language model's output. 
#It affects how creative or deterministic the responses are.
#Low Values  (0.0 – 0.3): More deterministicand predictable.
#High Values (0.7 – 1.5): More random, creative, and diverse.
#Recommended Temperature 
#Technical/Precise tasks          (math, code, facts)          0.0 - 0.2
#General conversation             (general QA, explanations)   0.5 - 0.7
#Creative writing, storytelling, jokes                         0.9 - 1.2
#Pure randomness (wild ideas, brainstorming)                   1.5+


model = ChatOpenAI(model='gpt-4',temperature=0)
result=model.invoke("Write a 5 line poem on cricket")
print(result.content)

#output will be : Cricket, a game of bat and ball,
#                 Under the sun, standing tall.
#                 Cheers echo as wickets fall,
#                 In this sport, loved by all.
#                 A sixer's flight, captivates us all.


model = ChatOpenAI(model='gpt-4',temperature=1.5)

#output will be : In the field under the sun's fiery spicket,
#                 Boys and girls with love for cricket,
#                 The ball tossed, a moment plays tetik,
#                 Victory's cheer or sunk wicket,
#                 Proving life is defined between any pitch and wicket.

model = ChatOpenAI(model='gpt-4',temperature=1.5, max_completion_tokens=10)

#output will be : With seams so narrow, field deep and wide, 
# as we can see the max_completion_token parameter limits the token usage 

#ek aur temperature parameter ka point ye hai ki if temperature 0 hoga to ek input ke liye 
# output same hi ayega no matter kitni baar bhi input bhejde 

# and if 0 ke ass pass hoga for eg 0.5 fir , same input pe baar baar output me hlke hlke changes ayenge 

# and if temperature 2 krdiya tb same input ke liye agr multiple times output request krenge to 
# hr baar output completely changed aayega 
