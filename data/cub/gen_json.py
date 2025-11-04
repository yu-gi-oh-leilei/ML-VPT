import json

def read_attributes_from_file(file_path):
    # Read lines from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Create a dictionary from the lines
    attributes_dict = {}
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            index = int(parts[0]) - 1  # Convert the index to zero-based
            key = parts[1]  # Get the descriptive part
            attributes_dict[key] = index
    return attributes_dict

def write_to_json(data, json_file_path):
    # Writing the dictionary to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Usage example
file_path = '/media/data0/MLICdataset/CUB_200_2011/attributes.txt'
json_file_path = 'attributes.json'
classname_path = 'classname.json'
# Processing the data
attributes_dict = read_attributes_from_file(file_path)
write_to_json(attributes_dict, json_file_path)

print("JSON file has been created with the following content:")
# print(attributes_dict)


attributes_dict = read_attributes_from_file(file_path)

write_to_json(list(attributes_dict.keys()), classname_path)

print(list(attributes_dict.keys()))


