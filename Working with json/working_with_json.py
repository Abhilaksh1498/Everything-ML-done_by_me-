# -*- coding: utf-8 -*-
"""
Created on Mon May 24 22:11:34 2021

@author: MSI_PC
"""
import json        # built-in

####### json objects can either be in string or a json file 
# json objects will load as python dict

# 1. You can parse a JSON string using json.loads() method
person = '{"name": "Bob", "languages": ["English", "Fench"]}'
person_dict = json.loads(person)

# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
print(type(person_dict), person_dict)

# Output: ['English', 'French']
print(person_dict['languages'])

# 2. You can use json.load() method to read a file containing JSON object
with open('person.txt', 'r') as j_object:       # .txt or .json both types will work
       data = json.load(j_object)
       
print(type(data), data)

# 3. You can convert a dictionary to JSON string using json.dumps() method.
person_dict = {'name': 'Bob',
'age': 12,
'children': None
}
person_json = json.dumps(person_dict)

# Output: <'str'> , {"name": "Bob", "age": 12, "children": null}
print(type(person_json), person_json)

# 4. To write JSON to a file in Python, we can use json.dump() method
with open('person_trial.json', 'w') as json_file:      # couldbe a .txt file too
  json.dump(person_dict, json_file)