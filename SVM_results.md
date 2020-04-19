# Results for the SVM assignement
For a cross-validation done by ```GridSearchCV``` from ```sklearn```, the following accuracies were got:

```params``` | mean accuracy (```mean_test_score```)
--- | ---
{'C': 1e-07, 'gamma': 1000, 'kernel': 'linear'} | 0.8987999999999999
{'C': 1e-07, 'gamma': 1000, 'kernel': 'rbf'} | 0.1136
{'C': 1e-07, 'gamma': 100, 'kernel': 'linear'} | 0.8987999999999999
{'C': 1e-07, 'gamma': 100, 'kernel': 'rbf'} | 0.1136
{'C': 1e-07, 'gamma': 0.1, 'kernel': 'linear'} | 0.8987999999999999
{'C': 1e-07, 'gamma': 0.1, 'kernel': 'rbf'} | 0.1136
{'C': 0.1, 'gamma': 1000, 'kernel': 'linear'} | 0.9032
{'C': 0.1, 'gamma': 1000, 'kernel': 'rbf'} | 0.1136
{'C': 0.1, 'gamma': 100, 'kernel': 'linear'} | 0.9032
{'C': 0.1, 'gamma': 100, 'kernel': 'rbf'} | 0.1136
{'C': 0.1, 'gamma': 0.1, 'kernel': 'linear'} | 0.9032
{'C': 0.1, 'gamma': 0.1, 'kernel': 'rbf'} | 0.1136
{'C': 100000, 'gamma': 1000, 'kernel': 'linear'} | 0.9032
{'C': 100000, 'gamma': 1000, 'kernel': 'rbf'} | 0.1136
{'C': 100000, 'gamma': 100, 'kernel': 'linear'} | 0.9032
{'C': 100000, 'gamma': 100, 'kernel': 'rbf'} | 0.1136
{'C': 100000, 'gamma': 0.1, 'kernel': 'linear'} | 0.9032
{'C': 100000, 'gamma': 0.1, 'kernel': 'rbf'} | 0.1136

Where the optimized parameter values were:

```
SVC(C=0.1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1000, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

Accuracy on the test set with the optimized parameter values: 0.9
