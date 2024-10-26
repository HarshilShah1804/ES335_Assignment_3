import re
import csv

def tokenize_code(data):
    # Define the regular expression pattern to match various elements
    pattern = r'(""")|(\n)|(\t)|(\s{4,})|(\w+\s?)|([^\w\s])'
    # (""")|(\n)|(\s{4})|([^\w\s])|(\w+)

    # Find all tokens based on the pattern
    tokens = re.findall(pattern, data)

    # Flatten the list of tuples and filter out None values
    tokens = [token for group in tokens for token in group if token]

    points_to_check = []
    for i in range(len(tokens)):
        token = tokens[i]
        if(len(token) > 2 and token[-1] == "\n"):
            points_to_check.append(i)
            tokens[i] = token[:-1]
    
    counter = 0
    for index in points_to_check:
        tokens.insert(index+1+counter, "\n")
        counter += 1
        
    return tokens

with open ("dataset/refined_data.txt", "r", encoding='utf-8') as file:
    data = re.split('----------', file.read())

    # generate tokens
    tokens = []
    for i in range(len(data)):
        tokens.extend(tokenize_code(data[i]))

    make_unique = sorted(set(tokens))
    make_unique.insert(0, '')

    # assign unique ids to each token
    token_to_id = {}
    for i in range(len(make_unique)):
        token_to_id[make_unique[i]] = i
    # token_to_id[''] = 0
    # with open("dataset/tokens.csv", "w") as file:
    #     writer = csv.writer(file)
    #     for key, value in token_to_id.items():
    #         writer.writerow([key, value])
    # print(len(token_to_id))
    # print(token_to_id['.'])

    # write to csv, in block of 5 context length
    with open("D:/IIT Gandhinagar/Sem 3/ML/dataset_20.csv", "w") as file:
        writer = csv.writer(file)
        for line in data:
            tokens = tokenize_code(line)
            block = [0] * (20 + 1) # context length = 20
            for i in range(len(tokens)):
                # print(tokens[i])
                block = block[1:] + [token_to_id[tokens[i]]]
                writer.writerow(block)

