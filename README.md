# ml-classification

## The Problem
In real life, most of the time is spent on finding variable to populate your database and feature engineering. However, finding the best model to fit your problem is equally important and can be tricky if you don't know what parameters to use. In this project, I try to come up with a simple system to try different parameters for different models and automatically find the best so you don't have to manually iterate them and potencially miss the target.

## Dataset
The dataset consists of a collection of quantitative features of red wines. The class, quality, represents the score each wine received. It totals 1,598 entries and 12 columns including the class. There are no missing values.

The quality of the wine ranges from 3 up to 8. Supposing I am a wine enthusiast who is willing to always find the best possible wines, the quality class will be split into two: scores 7 and 8 will be classified as good (1) and the rest will be classified as bad (0). At the end, bad quality wine totals for 86.4% of the class and good quality wine 13.6%.

Dataset link: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009


## Preprocessing
The main goal of this project are the models, therefore less time will be spent on preprocessing and feature engineering will not be performed. Firstly, the dataset will be split into 80% train and 20% test. As the quality transformation results in a umbalanced class, the train set will be rebalanced making use of the SMOTE technique, creating synthetic instances until the minority class has the same number of examples as the majority class. Finally, both train and test sets will be scaled before fed into the models.


## Modelling and Evaluation
In order to better illustrate what I'm trying to achieve, 6 models were chosen for the classification: Decision Tree, Gradient Boosting, k-NN, Logistic Regression, Random Forest and Support Vector Classifier (SVC). Supposing I'm only buying wine for consumption, I want them to be good as much as possible. In other words, I want to be right about the 1 class as many times as possible even if it means not finding all the good wines out there. In conclusion, what I'm looking for here is a model with the highest precision score.

Using the k-NN model as an example, I want to find the number of neighbors that best fit my model in terms of its precision score. As I don't know which `n_neighbors` to choose, the code below iterates for different values of n and plots the results:

``` PYTHON
scores = []
n_neighbors = []

n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

for i in n:    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    score = precision_score(y_test, y_pred)
    scores.append(score)
    n_neighbors.append([i, score])

plt.figure(figsize=(10,8))
plt.title("n Neighbors vs Precision")
plt.xlabel("n Neighbors")
plt.ylabel("Precision")
pd.Series(scores, n).plot(kind='bar', color='#50589F')
```

![knn_example](https://github.com/andrepelegrini/ml-classification/issues/1#issue-1291394339)

As can be seen, the model derives the best precision score when `n_neighbors=2`.

The same logic is applied to all 6 different models with different parameters.

## Conclusion
Although the dataset preparation, feature engineering and preprocessing amounts for most of the time spent on a machine learning problem, it's nice to have a few hacks that can help you choose the best model. However, in order to best apply them, it's important to bear in mind a few things:

1. **Know your model**: not all parameters are worth iterating and not all of them can be left out. Get comfortable with them and evaluate each ones to choose and why;

2. **Choose the best metrics**: rest assured you'll probably never encounter a problem which results 1 in every single metric. In a classification problem, there are different metrics to evaluate, such as precision, recall, f1-score and accuracy. Know them in detail and evaluate which fits best your problem. In the example above, a wine drinker doesn't want to find all good wine in the world, she just wants to drink good wine whenever she buys one. But suppose you want to sell wine. In that case, the approach might be finding all best wine even if it means bringing in some bad ones. Therefore, recall may be a good option. Is it possible to find a model with both precision and recall scoring high? Perhaps, but most of the times you'll have to make a choice, that's why it's a called a tradeoff. In any case, knowing your problem and metrics goes a long way.


