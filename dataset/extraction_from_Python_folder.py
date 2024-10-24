import os

def extract_python_files(root_folder, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.endswith('.py'):
                    filepath = os.path.join(foldername, filename)
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                        outfile.write('\n' + '-' * 10 + '\n')

# Usage
root_folder = 'dataset\Python'  # Replace with your folder path if needed
output_file = 'all_python_files.txt'
extract_python_files(root_folder, output_file)

print(f"All Python files have been extracted into '{output_file}'.")
