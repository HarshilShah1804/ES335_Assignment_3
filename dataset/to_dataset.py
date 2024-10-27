import re
import csv

def tokenize_code(data):
    # Updated regular expression pattern:
    pattern = r"(\b\w+\b|\d+|[^\w\s]|\s+)"

    # Explanation:
    # (\d+|\w+)       - Match numbers or words
    # ([.,;:?!()\[\]{}"\']) - Match special symbols
    # (\s+)           - Match any whitespace (including spaces, tabs, newlines)

    # Find all tokens based on the pattern
    tokens = re.findall(pattern, data)

    # Flatten the list of tuples and filter out None values
    # tokens = [token for group in tokens for token in group if token]

    return tokens

with open("sherlock_holmes.txt", "r") as file:
    data = re.split('\n', file.read())

    # for line in data[:12]:
    #     if line == '':
    #         continue
    #     print(tokenize_code(line))

    # Generate tokens
    tokens = []
    for i in range(len(data)):
        if data[i] == "":
            continue
        tokens.extend(tokenize_code(data[i]))

    # Create a sorted unique list of tokens
    make_unique = sorted(set(tokens))
    make_unique.insert(0, '')  # Add an empty token at index 0
    make_unique.insert(1, '<UNK>')  # Add an unknown token at index 1

    # Assign unique IDs to each token
    token_to_id = {token: i for i, token in enumerate(make_unique)}

    # Write token-to-ID mapping to tokens.csv
    with open("tokens_holmes.csv", "w") as file:
        writer = csv.writer(file)
        for key, value in token_to_id.items():
            writer.writerow([key, value])

    # Write the tokenized dataset to dataset.csv in blocks of 20
    with open("dataset_holmes_15.csv", "w") as file:
        writer = csv.writer(file)
        for line in data:
            tokens = tokenize_code(line)
            block = [0] * (15 + 1)  # 20 context length + 1 for the current token
            for i in range(len(tokens)):
                block = block[1:] + [token_to_id[tokens[i]]]
                writer.writerow(block)
            writer.writerow(block[1:] + [0])  # End block with padding
