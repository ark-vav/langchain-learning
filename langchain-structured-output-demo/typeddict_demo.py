from typing import TypedDict

class Person(TypedDict) :

    name : str
    age : int 

new_person: Person = { 'name':'vaibhav','age':'35'}

print(new_person)

# TypedDict defines the structure of the graph's state.
# It specifies what keys exist in the state dictionary and what type of data each key stores.

# output : {'name': 'vaibhav', 'age': '35'} 