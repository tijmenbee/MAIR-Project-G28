from baseline_majority import BaselineMajority
from baseline_rulebased import BaselineRuleBased

from data import train_data, dev_data, deduped_train_data, deduped_dev_data
from decision_tree import DecisionTree
from feedforward_nn import FeedForwardNN
from logistic_regression import LogisticRegressionModel
from rulebased_nn_merge import RuleBasedNN

TRAINING = True


if TRAINING:
    print("Evaluating on DEV set.")
else:
    print("Evaluating on TEST set. DO NOT KEEP DOING THIS. ONLY FOR THE FINAL STEP.")
print()


def test_model_accuracy(model, model_name: str, deduped=False):
    from data import test_data, deduped_test_data
    from collections import Counter
    if TRAINING:
        testing_data = dev_data if not deduped else deduped_dev_data
    else:
        testing_data = test_data if not deduped else deduped_test_data

    test_sentences = [sentence for act, sentence in testing_data]
    test_acts = [act for act, sentence in testing_data]

    pred_acts = model.predict(test_sentences)

    FP = []
    TP = []
    FN = []
  
    acts = Counter(test_acts)
    for i in range(len(test_acts)):
        if pred_acts[i] == test_acts[i]:
            TP.append(test_acts[i])
        else:
            FP.append(test_acts[i])
            FN.append(pred_acts[i])

    TP = Counter(TP)
    FN = Counter(FN)
    FP = Counter(FP)
    
    correct = sum(pred_act == test_act for pred_act, test_act in zip(pred_acts, test_acts))

    F1_dict = {}
    for act in list(acts):
        if TP[act] == 0:
            precision = 0
            recall = 0
            F1 = 0
        else:
            precision = TP[act]/(TP[act]+FP[act])
            recall = TP[act]/(TP[act]+FN[act])
            F1 = 2*(precision*recall)/(precision+recall)
            accuracy = TP[act]/(TP[act]+FN[act]+FP[act])
        #print(act, ", precision: ", precision,  ", recall: ", recall,  ", F1: ", F1,  ", accuracy: ", accuracy)
        F1_dict[act] = F1
    weighted_f1 = sum(F1_dict[key]*acts.get(key, 0) for key in acts) / len(test_acts)

    print(f"{model_name} accuracy: {correct / len(testing_data) * 100:.1f}% , weighted F1: {weighted_f1} ({model.info})")


def test_model_precision(model, model_name: str, deduped=False):
    from data import test_data, deduped_test_data
    if TRAINING:
        testing_data = dev_data if not deduped else deduped_dev_data
    else:
        testing_data = test_data if not deduped else deduped_test_data

# TODO evaluate on precision, recall, f1?


train_acts = [act for act, _ in train_data]
train_sentences = [sentence for _, sentence in train_data]

deduped_train_acts = [act for act, _ in deduped_train_data]
deduped_train_sentences = [sentence for _, sentence in deduped_train_data]


baseline_majority = BaselineMajority(train_acts)
test_model_accuracy(baseline_majority, "BaselineMajority")

baseline_rulebased = BaselineRuleBased(train_acts)
test_model_accuracy(baseline_rulebased, "BaselineRuleBased")

feedforward_nn = FeedForwardNN(train_data, dev_data)
test_model_accuracy(feedforward_nn, "FeedForwardNN")

deduped_feedforward_nn = FeedForwardNN(deduped_train_data, deduped_dev_data, epochs=4)
test_model_accuracy(deduped_feedforward_nn, "DedupedFeedForwardNN", deduped=True)

rulebased_nn_merged = RuleBasedNN(deduped_train_data, deduped_dev_data, epochs=4)
test_model_accuracy(rulebased_nn_merged, "DedupedRuleBasedNN", deduped=True)

logistic_regression = LogisticRegressionModel(train_data)
test_model_accuracy(logistic_regression, "LogisticRegressionModel")

deduped_logistic_regression = LogisticRegressionModel(deduped_train_data)
test_model_accuracy(deduped_logistic_regression, "DedupedLogisticRegressionModel", deduped=True)

decision_tree = DecisionTree(train_data)
test_model_accuracy(decision_tree, "DecisionTree")

deduped_decision_tree = DecisionTree(deduped_train_data)
test_model_accuracy(deduped_decision_tree, "DedupedDecisionTree", deduped=True)
