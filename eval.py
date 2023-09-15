from baseline_majority import BaselineMajority
from baseline_rulebased import BaselineRuleBased

from data import train_data, dev_data, deduped_train_data, deduped_dev_data
from feedforward_nn import FeedForwardNN
from logistic_regression import LogisticRegressionModel

TRAINING = True


if TRAINING:
    print("Evaluating on DEV set.")
else:
    print("Evaluating on TEST set. DO NOT KEEP DOING THIS. ONLY FOR THE FINAL STEP.")
print()


def test_model(model, model_name: str, deduped=False):
    from data import test_data, deduped_test_data

    if TRAINING:
        testing_data = dev_data if not deduped else deduped_dev_data
    else:
        testing_data = test_data if not deduped else deduped_test_data

    test_sentences = [sentence for act, sentence in testing_data]
    test_acts = [act for act, sentence in testing_data]

    pred_acts = model.predict(test_sentences)

    correct = sum(pred_act == test_act for pred_act, test_act in zip(pred_acts, test_acts))

    print(f"{model_name} accuracy: {correct / len(testing_data) * 100:.1f}% ({model.info})")

    del test_data  # No leakage!
    del deduped_test_data


train_acts = [act for act, _ in train_data]
train_sentences = [sentence for _, sentence in train_data]

deduped_train_acts = [act for act, _ in deduped_train_data]
deduped_train_sentences = [sentence for _, sentence in deduped_train_data]


baseline_majority = BaselineMajority(train_acts)
test_model(baseline_majority, "BaselineMajority")

baseline_rulebased = BaselineRuleBased(train_acts)
test_model(baseline_rulebased, "BaselineRuleBased")

feedforward_nn = FeedForwardNN(train_data, dev_data)
test_model(feedforward_nn, "FeedForwardNN")

deduped_feedforward_nn = FeedForwardNN(deduped_train_data, deduped_dev_data, epochs=4)
test_model(deduped_feedforward_nn, "DedupedFeedForwardNN", deduped=True)

logistic_regression = LogisticRegressionModel(train_data)
test_model(logistic_regression, "LogisticRegressionModel")

deduped_logistic_regression = LogisticRegressionModel(deduped_train_data)
test_model(deduped_logistic_regression, "DedupedLogisticRegressionModel", deduped=True)
