from data import load_ASERTAIN
from utils import print_log

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import json


def train(selected_modalities, label,  model, trials, test_size):
    acc = 0
    f1 = 0
    for i in range(trials):
        print_log("trial: " + str(i))
        X, y, _, _, _, _, _, _, _ = load_ASERTAIN(selected_modalities[0], label, 80, 10, 10, i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        y_pred = model.fit(X_train, y_train).predict(X_test)
        acc += accuracy_score(y_test, y_pred)
        f1 += f1_score(y_test, y_pred, average='macro')


    
    print(acc/trials, f1/trials)


if __name__ == "__main__":

    # Open and read the JSON configuration file
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    selected_modalities = config['modalities']
    label = config['label']
    if config['model'] == "svm":
        model = svm.SVC()
    else:
        model = GaussianNB()

    trials = config['trials']
    test_size = config['test_size']

    train(selected_modalities, label, model, trials, test_size)