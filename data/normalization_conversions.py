import pandas as pd

train_x_old = pd.read_csv('data/train_x_old.csv')
print(train_x_old["Total GHI"].mean())