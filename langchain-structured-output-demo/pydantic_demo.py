# Pydantic is a data validation and data parsing library for Python. 
# It ensures that the data you work with is correct, and type-safe.



from pydantic import BaseModel, EmailStr, Field 
from typing import Optional

class Student(BaseModel):

    name: str = 'vav'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value' \
    'representing the cgpa of the student')

new_student = {'age':32,'email':'abc@gmail.com'}

student = Student(**new_student)

student_dict = dict(student)

print(student_dict['age'])

student_json = student.model_dump_json()