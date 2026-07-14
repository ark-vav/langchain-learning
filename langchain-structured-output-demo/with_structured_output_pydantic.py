#USING PYDANTIC 

from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv 
from typing import Optional, Literal
from pydantic import BaseModel, Field 


load_dotenv()

model = ChatOpenAI()

# schema 
class Review(BaseModel):

    key_themes: list[str]= Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos","neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")


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
Expensive compared to competition 

Review by Vaibhav Ghildiyal                                  
""") 

print(result.name)  

# output : Vaibhav Ghildiyal 

print(result) #to view full pydantic object 

#output : 
# Review(
#     key_themes=[
#         'Performance',
#         'Battery Life',
#         'Fast Charging',
#         'Camera Quality',
#         'S-Pen',
#         'Design',
#         'One UI',
#         'Bloatware',
#         'Pricing'
#     ],
#     summary='The review highly praises the Samsung Galaxy S24 Ultra for its exceptional performance, camera, battery life, fast charging, and S-Pen support, while criticizing its bulky design, One UI bloatware, and premium price.',
#     sentiment='pos',
#     pros=[
#         'Powerful Snapdragon 8 Gen 3 processor',
#         'Excellent 200MP camera with impressive zoom',
#         'Long-lasting 5000mAh battery',
#         '45W fast charging',
#         'Useful S-Pen support'
#     ],
#     cons=[
#         'Bulky and heavy for one-handed use',
#         'One UI includes unnecessary bloatware',
#         'Expensive compared to competitors'
#     ],
#     name='Vaibhav Ghildiyal'
# )
#
# Explanation:
# - Since Review inherits from Pydantic's BaseModel, the output is a Review object.
# - You can access each field as an attribute:
#     result.name
#     result.summary
#     result.sentiment
#     result.pros
#     result.cons
# - To convert the object into a Python dictionary, use:
#     result.model_dump()
# - To convert it into a JSON string, use:
#     result.model_dump_json(indent=4)


# ======================== PROGRAM SUMMARY ========================
#
# 1. Loads environment variables using dotenv (loads the OpenAI API key).
#
# 2. Creates an OpenAI chat model using ChatOpenAI().
#
# 3. Defines a structured output schema using a Pydantic BaseModel named 'Review'.
#    The schema specifies exactly what information the LLM should return:
#    - key_themes : List of main topics discussed in the review.
#    - summary    : Short summary of the review.
#    - sentiment  : Must be either "pos" or "neg" (Literal restricts valid values).
#    - pros       : Optional list of advantages.
#    - cons       : Optional list of disadvantages.
#    - name       : Optional reviewer name.
#
# 4. Uses model.with_structured_output(Review) to bind the schema to the LLM.
#    This forces the model to return a Review object instead of plain text.
#
# 5. Sends a Samsung Galaxy S24 Ultra review to the LLM using invoke().
#
# 6. The model extracts the requested information according to the schema and
#    returns a Review (Pydantic) object.
#
# 7. Prints only the reviewer's name using:
#       print(result.name)
#
# Expected Output:
# Vaibhav Ghildiyal
#
# Key Concepts Learned:
# - Pydantic BaseModel creates structured schemas.
# - Field() provides descriptions and default values for each field.
# - Literal restricts a field to specific allowed values.
# - Optional allows a field to be omitted (None).
# - with_structured_output() forces the LLM to return structured data matching
#   the Pydantic schema instead of free-form text.
#
# ===============================================================
