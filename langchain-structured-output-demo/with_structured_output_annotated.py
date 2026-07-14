#USING TYPEDDICT 

from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv 
from typing import TypedDict, Annotated , Optional, Literal

load_dotenv()

model = ChatOpenAI()

# schema 
class Review(TypedDict):

    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str,"A brief summary of the review"]
    sentiment: Annotated[Literal["pos","neg"],"Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! 
The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. 
The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. 
What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. 
Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use.
Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? 
The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy—not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors""") 

print(result) 

# output : 

"""
Output:
{
    'key_themes': [
        'Performance',
        'Battery Life',
        'Fast Charging',
        'Camera Quality',
        'S-Pen',
        'Design',
        'One UI',
        'Bloatware',
        'Pricing'
    ],
    'summary': 'The review highly praises the Samsung Galaxy S24 Ultra for its exceptional performance, camera, battery life, fast charging, and S-Pen support, while criticizing its bulky design, One UI bloatware, and premium price.',
    'sentiment': 'positive',
    'pros': [
        'Powerful Snapdragon 8 Gen 3 processor',
        'Excellent 200MP camera with impressive zoom',
        'Long-lasting 5000mAh battery',
        '45W fast charging',
        'Useful S-Pen support'
    ],
    'cons': [
        'Bulky and heavy for one-handed use',
        'One UI includes unnecessary bloatware',
        'Expensive compared to competitors'
    ]
}

Explanation:
- with_structured_output(Review) forces the LLM to return data matching the Review TypedDict schema.
- The output is a Python dictionary, NOT an AIMessage.
- The Annotated descriptions help the LLM understand what each field should contain.
- The exact wording may vary between runs, but the structure (keys and data types) remains the same.
"""
