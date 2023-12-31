from sklearn.model_selection import train_test_split


def extract_data():
    with open('data/raw_data/dialog_acts.dat', 'r') as f:
        dialog_acts = f.readlines()

    data = [tuple(line.lower().strip().split(" ", maxsplit=1)) for line in dialog_acts]

    ACTS = list(set([act for act, _ in data]))

    train_data, test_data = train_test_split(data, test_size=0.1)
    train_data, dev_data = train_test_split(train_data, test_size=0.15)

    # Now, using data, we remove all duplicates
    deduped_data = list(set(data))

    deduped_train_data, deduped_test_data = train_test_split(deduped_data, test_size=0.15)

    deduped_train_data, deduped_dev_data = train_test_split(deduped_train_data, test_size=0.15)

    return ACTS, train_data, dev_data, test_data, deduped_train_data, deduped_dev_data, deduped_test_data


# To then be used in other files:
ACTS, train_data, dev_data, test_data, deduped_train_data, deduped_dev_data, deduped_test_data = extract_data()
