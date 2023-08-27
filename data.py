import os
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
import csv
from itertools import groupby
import random
from sklearn.impute import SimpleImputer



DATA_DIR = os.path.join(os.path.dirname(__file__), 'datasets')
ASCERTAIN_DIR = os.path.join(DATA_DIR, 'ASCERTAIN_Features')
ASCERTAIN_FILE = os.path.join(ASCERTAIN_DIR, 'ascertain_multimodal.csv')

def load_ASERTAIN(selected_modalities=['ECG', 'GSR'],  label='valence', train_ratio=60, val_ratio=20, test_ratio=20, trial=0):

    # Change seed for each trial
    random.seed(trial)
    
    n_subjects = 58
    n_cases = 36
    n_traits = 5

    if label == 'valence':
        label_index = 3
    elif label == 'arousal':
        label_index = 2

	# Convert csv to np array
    with open(ASCERTAIN_FILE) as file:
        reader = csv.reader(file)
        data = list(reader)
        columns = np.asarray(data[0])
        data = np.asarray(data[1:]).astype(float)

    selected_modalities.append('Personality')
    
    # Select index of selected features, personality, labels and ids
    selected_index = select_idex(columns, selected_modalities)
    
    data = data[:,selected_index]

    """
    split train and test dataset subject-wise based on self.split_ratio
    group data upon subject id
    """
    data_grouped = [list(it) for k, it in groupby(data.tolist())]
    random.shuffle(data_grouped)


    subject_attributes = []
    video_attributes = []
    low_personality_attributes = []
    high_personality_attributes = []

    for i in range(n_subjects):
        subject_id = []
        subject_attributes.append(subject_id)

    for i in range(n_cases):
        video_id = []
        video_attributes.append(video_id)

    for i in range(n_traits):
        personality_id = []
        low_personality_attributes.append(personality_id)
        high_personality_attributes.append(personality_id)


    all_data = [item for sublist in data_grouped for item in sublist]
    all_data = np.asarray(all_data)
    

    for i in range(len(all_data[0:, 4:])):
        subject_id = int(all_data[i][0])
        subject_attributes[subject_id].append(i)
        case_id = int(all_data[i][1])
        video_attributes[case_id].append(i)
        for j in range(n_traits):
            personality_trait = all_data[i][all_data.shape[1]-j-1]
            if j == 0 or j == 3:
                threshold = 4
            else:
                threshold = 5
            if personality_trait < threshold:
                low_personality_attributes[j].append(i)
            else:
                high_personality_attributes[j].append(i)

    # Remove ids, labels and personalities from features
    X = all_data[0:, 4:all_data.shape[1]-5]
    y = all_data[0:, label_index]


    train_mask, test_mask, valid_mask = generate_masks(X, train_ratio, test_ratio)

    X = preprocessing(X, y, train_mask, valid_mask, test_mask)

    return X, y, train_mask, test_mask, valid_mask, subject_attributes, video_attributes, low_personality_attributes, high_personality_attributes


def is_column_feature(columns, column_index):
	return ('label' not in columns[column_index] and 'id' not in columns[column_index])

def scale_data(X, train_mask, valid_mask, test_mask, y):
    scaler = StandardScaler()
    X[train_mask] = scaler.fit_transform(X[train_mask], y[train_mask])
    X[valid_mask] = scaler.transform(X[valid_mask])
    X[test_mask] = scaler.transform(X[test_mask])
    return X

def inf_remover(X, train_mask, valid_mask, test_mask, y):
    impInf = SimpleImputer(missing_values=np.inf, strategy='mean')
    X[train_mask] = impInf.fit_transform(X[train_mask], y[train_mask])
    X[valid_mask] = impInf.transform(X[valid_mask])
    X[test_mask] = impInf.transform(X[test_mask])
    return X

def nanmean_labelwise(X, y):
    # Replace nan values with mean of columns sharing the same label
    label_means = {}
    unique_labels = np.unique(y)
    for label in unique_labels:
        label_means[label] = np.nanmean(X[y == label], axis=0)

    for i in range(X.shape[0]):
        label = y[i]
        mask = np.isnan(X[i])
        X[i, mask] = label_means[label][mask]

    return X

def generate_masks(X, train_ratio, test_ratio):
    train_mask = [True for i in range(round(len(X)*train_ratio/100))] + [False for i in range(round(len(X)-len(X)*train_ratio/100))]
    test_mask = [False for i in range(round(len(X) - len(X)*test_ratio/100))] + [True for i in range(round(len(X)*test_ratio/100))]
    valid_mask =  np.logical_and(np.logical_not(train_mask),  np.logical_not(test_mask))
    return train_mask, test_mask, valid_mask

def preprocessing(X, y, train_mask, valid_mask, test_mask):
    X = nanmean_labelwise(X, y)
    X = np.nan_to_num(X)
    X = normalize(X)
    X = inf_remover(X, train_mask, valid_mask, test_mask, y)
    X = scale_data(X, train_mask, valid_mask, test_mask, y)
    return X

def select_idex(columns, selected_modalities):
    return [i for i in range(len(columns)) if (not is_column_feature(columns, i)) or ((is_column_feature(columns, i) and (columns[i].split('_')[0] in selected_modalities))) ]


if __name__ == "__main__":
    pass