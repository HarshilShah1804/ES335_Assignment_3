import pandas as pd
csv_file_path = 'dataset/archive (1)/train.csv'
data = pd.read_csv(csv_file_path)
print(data.head())