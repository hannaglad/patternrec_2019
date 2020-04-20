# Results for the SVM assignement
(RBF kernel on subset of data) 
For a cross-validation done by ```GridSearchCV``` from ```sklearn```, the following accuracies were got:

```params``` | mean accuracy (```mean_test_score```)
--- | ---
{'C': 0.1, 'gamma': 1e-07, 'kernel': 'rbf'} | 0.86525
{'C': 0.1, 'gamma': 1e-05, 'kernel': 'rbf'} | 0.1125
{'C': 0.1, 'gamma': 0.001, 'kernel': 'rbf'} | 0.1125
{'C': 0.1, 'gamma': 0.1, 'kernel': 'rbf'} | 0.1125
{'C': 1, 'gamma': 1e-07, 'kernel': 'rbf'} | 0.9202499999999999
{'C': 1, 'gamma': 1e-05, 'kernel': 'rbf'} | 0.17525
{'C': 1, 'gamma': 0.001, 'kernel': 'rbf'} | 0.1125
{'C': 1, 'gamma': 0.1, 'kernel': 'rbf'} | 0.1125
{'C': 10, 'gamma': 1e-07, 'kernel': 'rbf'} | 0.9315
{'C': 10, 'gamma': 1e-05, 'kernel': 'rbf'} | 0.17925
{'C': 10, 'gamma': 0.001, 'kernel': 'rbf'} | 0.1125
{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} | 0.1125
{'C': 100, 'gamma': 1e-07, 'kernel': 'rbf'} | 0.9315000000000001
{'C': 100, 'gamma': 1e-05, 'kernel': 'rbf'} | 0.17925
{'C': 100, 'gamma': 0.001, 'kernel': 'rbf'} | 0.1125
{'C': 100, 'gamma': 0.1, 'kernel': 'rbf'} | 0.1125
{'C': 1000, 'gamma': 1e-07, 'kernel': 'rbf'} | 0.9315000000000001
{'C': 1000, 'gamma': 1e-05, 'kernel': 'rbf'} | 0.17925

Where the optimized parameter values were:

```
SVC(C=0.001, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=2, gamma=0.1, kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

Accuracy on the test set with the optimized parameter values: 1.0
