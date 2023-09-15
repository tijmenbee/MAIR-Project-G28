from sklearn.model_selection import train_test_split

ACTS = [
    "ack",
    "affirm",
    "bye",
    "confirm",
    "deny",
    "hello",
    "inform",
    "negate",
    "null",
    "repeat",
    "reqalts",
    "reqmore",
    "request",
    "restart",
    "thankyou",
]

with open('../dialog_acts.dat', 'r') as f:
    dialog_acts = f.readlines()

data = [tuple(line.lower().strip().split(" ", maxsplit=1)) for line in dialog_acts]

train_data, test_data = train_test_split(data, test_size=0.15, random_state=42)

train_data, dev_data = train_test_split(train_data, test_size=0.15, random_state=42)

# Now, using data, we remove all duplicates
deduped_data = list(set(data))

deduped_train_data, deduped_test_data = train_test_split(deduped_data, test_size=0.15, random_state=42)

deduped_train_data, deduped_dev_data = train_test_split(deduped_train_data, test_size=0.15, random_state=42)

print(len(data))
print(len(deduped_data))

del data  # To avoid misusing original data.
