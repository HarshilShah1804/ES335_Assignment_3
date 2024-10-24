"""
"Here is an example of a nested loop in Python to print every combination of numbers between 0-9, excluding any combination that contains the number 5 or repeating digits:

```python
for i in range(10):  # First digit
    for j in range(10):  # Second digit
        for k in range(10):  # Third digit
            # Checking for the conditions
            if i != 5 and j != 5 and k != 5 and i != j and i != k and j != k:
                print(i, j, k)
```

This code will generate and print every combination of three digits between 0-9 that do not contain the number 5 and do not have any repeating digits." "Create a nested loop to print every combination of numbers between 0-9, excluding any combination that contains the number 5. Additionally, exclude any combination that contains a repeating digit. Implement the solution without using any built-in functions or libraries to check for repeating digits." "You are a Python code analyst, evaluating scripts for potential improvements."

"""

import re

with open ('data_raw.txt') as file:
    # all the is inclosed in "" so separate the data by " and then remove the empty strings
    data = re.split('" "|"\n"', file.read())
    # write this data in another file called refined_data.txt
    # for i in range(0, 13):
    #     print(data[i], i)
    with open('refined_data.txt', 'w') as refined_file:
        for line in range(0, len(data), 3):
            #format the data in the required format
            split1 = data[line].split('\n\n```python')
            if len(split1) < 2:
                continue
            split2 = split1[1].split('```\n\n')
            if len(split2) < 2:
                continue
            data[line] = '"""\n' + split1[0] + split2[1] + '\n"""\n' + split2[0]
            refined_file.write(data[line] + '\n')
            refined_file.write("----------" + '\n')