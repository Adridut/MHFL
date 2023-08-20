from data import load_ASERTAIN
from utils import print_log


from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score



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

    #Params
    selected_modalities = [['ECG', 'GSR', 'EMO', 'EEG']]
    label = 'valence'
    model = svm.SVC()
    # model = GaussianNB()
    trials = 10
    test_size = 0.2

    train(selected_modalities, label, model, trials, test_size)