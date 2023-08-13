import pandas as pd

def read_file(path):
    file = pd.read_csv(path)
    file = file.to_numpy()
    return file
