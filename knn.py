import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score

def knn_features(X_train, X_test, y_train, y_test):

    n_neighbors = []
    for i in range(1,10):    
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = precision_score(y_test, y_pred)
        n_neighbors.append([i, score])
        
    id_n = pd.DataFrame(n_neighbors, columns=['n_neighbors', 'score'])['score'].idxmax()
    n = pd.DataFrame(n_neighbors, columns=['n_neighbors', 'score'])['feature'][id_n]
        
    return n