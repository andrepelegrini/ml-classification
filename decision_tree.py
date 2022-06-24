import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score

def decision_trees_features(X_train, X_test, y_train, y_test):

    depths = []
    max_depths = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    for i in max_depths:
        model = DecisionTreeClassifier(max_depth=i, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = precision_score(y_test, y_pred)
        depths.append([i, score])
        
    id_depth = pd.DataFrame(depths, columns=['feature', 'score'])['score'].idxmax()
    max_depth = pd.DataFrame(depths, columns=['feature', 'score'])['feature'][id_depth]
    
    features = []
    max_features = ['auto', None, 'sqrt', 0.95, 0.75, 0.5, 0.25, 0.10]
    for i in max_features:
        model = DecisionTreeClassifier(max_features=i, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = precision_score(y_test, y_pred)
        features.append([i, score])

    id_feature = pd.DataFrame(features, columns=['feature', 'score'])['score'].idxmax()
    max_features = pd.DataFrame(features, columns=['feature', 'score'])['feature'][id_feature]
    
    leafs = []
    min_samples_leaf = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for i in min_samples_leaf:
        model = DecisionTreeClassifier(min_samples_leaf=i, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = precision_score(y_test, y_pred)
        leafs.append([i, score])
        
    id_leaf = pd.DataFrame(leafs, columns=['feature', 'score'])['score'].idxmax()
    min_samples_leaf = pd.DataFrame(leafs, columns=['feature', 'score'])['feature'][id_leaf]
        
    return max_depth, max_features, min_samples_leaf