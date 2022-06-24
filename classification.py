import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, recall_score, f1_score

from decision_tree import decision_trees_features
from knn import knn_features

df = pd.read_csv('winequality-red.csv')

def oversample_dataset(data):
    X = data.drop(columns={'quality'})
    y = data['quality']
    
    sm = SMOTE(random_state=42)
    
    X_sm, y_sm = sm.fit_resample(X, y)
    sm_df = pd.concat([X_sm, y_sm], axis=1)
    
    return sm_df

def preprocessing(data):

    data['quality'] = np.where(data['quality'] >= 7, 1, 0)

    X_train, X_test = train_test_split(data, test_size=0.20, random_state=42)

    X_train_sm = oversample_dataset(X_train)

    y_train = X_train_sm['quality']
    y_test = X_test['quality']
    X_train = X_train_sm.drop(columns={'quality'})
    X_test = X_test.drop(columns={'quality'})

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def run_models(data):
    
    X_train, X_test, y_train, y_test = preprocessing(data)

    max_depth_dt, max_features_dt, min_samples_leaf_dt = decision_trees_features(X_train, X_test, y_train, y_test)
    n_neighbors = knn_features(X_train, X_test, y_train, y_test)

    classifiers = {
    "Decision Tree": DecisionTreeClassifier(max_depth=max_depth_dt, 
                                            max_features=max_features_dt, 
                                            min_samples_leaf=min_samples_leaf_dt),
    "Support Vector Classifier": SVC(kernel='poly', 
                                     C=0.025),
    'NaiveBayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=n_neighbors),
    "Random Forest": RandomForestClassifier(n_estimators=500, 
                                            max_features=0.1, 
                                            min_samples_leaf=5, 
                                            n_jobs=-1)
}

    training = []
    testing = []
    precision = []
    recall = []
    f1 = []
    
    for key, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        y_test_pred = classifier.predict(X_test)

        training_score = cross_val_score(classifier, X_train, y_train, cv=5)
        training.append([classifier.__class__.__name__, round(training_score.mean(), 2)])

        testing_score = cross_val_score(classifier, X_test, y_test, cv=5)
        testing.append([classifier.__class__.__name__, round(testing_score.mean(), 2)])

        precision.append([classifier.__class__.__name__, round(precision_score(y_test, y_test_pred), 2)])
        recall.append([classifier.__class__.__name__, round(recall_score(y_test, y_test_pred), 2)])
        f1.append([classifier.__class__.__name__, round(f1_score(y_test, y_test_pred), 2)])
        
    metrics = {
        "training": training,
        "testing": testing,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return metrics

def fetch_best_model(data, metric):

    metrics = run_models(data)

    id_model = pd.DataFrame(metrics[metric], columns=['model', 'score'])['score'].idxmax()
    best_model = pd.DataFrame(metrics[metric], columns=['model', 'score'])['model'][id_model]

    return best_model


