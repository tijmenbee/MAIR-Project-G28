from baseline_majority import BaselineMajority
from baseline_rulebased import BaselineRuleBased

from data import train_data, dev_data
from giorgio_baselines.feedforward_nn import FeedForwardNN

TRAINING = True


if TRAINING:
    print("Evaluating on DEV set.")
else:
    print("Evaluating on TEST set. DO NOT KEEP DOING THIS. ONLY FOR THE FINAL STEP.")
print()


def test_model(model, model_name: str):
    from data import test_data

    if TRAINING:
        test_data = dev_data

    correct = 0
    for test_act, test_sentence in test_data:
        pred_act = model.predict(test_sentence)
        if pred_act == test_act:
            correct += 1

    print(f"{model_name} accuracy: {correct / len(test_data) * 100:.1f}% ({model.info})")

    del test_data  # No leakage!


def test_model_multiple(model, model_name: str):
    from data import test_data

    if TRAINING:
        test_data = dev_data

    test_sentences = [sentence for act, sentence in test_data]
    test_acts = [act for act, sentence in test_data]

    pred_acts = model.predict_multiple(test_sentences)

    correct = sum(pred_act == test_act for pred_act, test_act in zip(pred_acts, test_acts))

    print(f"{model_name} accuracy: {correct / len(test_data) * 100:.1f}% ({model.info})")

    del test_data  # No leakage!


train_acts = [act for act, _ in train_data]
train_sentences = [sentence for _, sentence in train_data]


baseline_majority = BaselineMajority(train_acts)
test_model(baseline_majority, "BaselineMajority")

baseline_rulebased = BaselineRuleBased(train_acts)
test_model(baseline_rulebased, "BaselineRuleBased")

feedforward_nn = FeedForwardNN(train_data, dev_data)
test_model_multiple(feedforward_nn, "FeedForwardNN")