from sklearn.model_selection import train_test_split

from baseline_majority import BaselineMajority
from baseline_rulebased import BaselineRuleBased

with open('../dialog_acts.dat', 'r') as f:
    dialog_acts = f.readlines()

data = [line.lower().strip().split(" ", maxsplit=1) for line in dialog_acts]
train_data, test_data = train_test_split(data, test_size=0.15, random_state=42)
del data  # To avoid misuse of non-training data


def test_model(model, model_name: str):
    correct = 0
    for test_act, test_sentence in test_data:
        pred_act = model.predict(test_sentence)
        if pred_act == test_act:
            correct += 1

    print(f"{model_name} accuracy: {correct / len(test_data) * 100:.1f}% ({model.info})")


train_acts = [act for act, _ in train_data]
train_sentences = [sentence for _, sentence in train_data]


baseline_majority = BaselineMajority(train_acts)
test_model(baseline_majority, "BaselineMajority")

baseline_rulebased = BaselineRuleBased(train_acts)
test_model(baseline_rulebased, "BaselineRuleBased")
