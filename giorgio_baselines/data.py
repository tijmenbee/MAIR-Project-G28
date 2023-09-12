from sklearn.model_selection import train_test_split

with open('../dialog_acts.dat', 'r') as f:
    dialog_acts = f.readlines()

data = [line.lower().strip().split(" ", maxsplit=1) for line in dialog_acts]

train_data, test_data = train_test_split(data, test_size=0.15, random_state=42)

del data  # To avoid misusing original data.
