import json

def read_json_file(filepath: str)-> dict:
    with open(filepath, "r") as input_file:
        return json.load(input_file)