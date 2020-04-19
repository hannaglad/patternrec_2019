# Results for the SVM assignement
For a cross-validation done by ```GridSearchCV``` from ```sklearn```, the following accuracies were got:

```params``` | mean accuracy (```mean_test_score```)
--- | ---
{'C': 1, 'gamma': 0.1, 'kernel': 'linear'} | 0.9032
{'C': 1, 'gamma': 0.1, 'kernel': 'poly'} | 0.9284000000000001
{'C': 1, 'gamma': 0.1, 'kernel': 'rbf'} | 0.1136
{'C': 1, 'gamma': 0.01, 'kernel': 'linear'} | 0.9032
{'C': 1, 'gamma': 0.01, 'kernel': 'poly'} | 0.9284000000000001
{'C': 1, 'gamma': 0.01, 'kernel': 'rbf'} | 0.1136
{'C': 1, 'gamma': 0.0001, 'kernel': 'linear'} | 0.9032
{'C': 1, 'gamma': 0.0001, 'kernel': 'poly'} | 0.9284000000000001
{'C': 1, 'gamma': 0.0001, 'kernel': 'rbf'} | 0.11480000000000001
{'C': 10, 'gamma': 0.1, 'kernel': 'linear'} | 0.9032
{'C': 10, 'gamma': 0.1, 'kernel': 'poly'} | 0.9284000000000001
{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} | 0.1136
{'C': 10, 'gamma': 0.01, 'kernel': 'linear'} | 0.9032
{'C': 10, 'gamma': 0.01, 'kernel': 'poly'} | 0.9284000000000001
{'C': 10, 'gamma': 0.01, 'kernel': 'rbf'} | 0.1136
{'C': 10, 'gamma': 0.0001, 'kernel': 'linear'} | 0.9032
{'C': 10, 'gamma': 0.0001, 'kernel': 'poly'} | 0.9284000000000001
{'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'} | 0.11520000000000001
{'C': 1000, 'gamma': 0.1, 'kernel': 'linear'} | 0.9032
{'C': 1000, 'gamma': 0.1, 'kernel': 'poly'} | 0.9284000000000001
{'C': 1000, 'gamma': 0.1, 'kernel': 'rbf'} | 0.1136
{'C': 1000, 'gamma': 0.01, 'kernel': 'linear'} | 0.9032
{'C': 1000, 'gamma': 0.01, 'kernel': 'poly'} | 0.9284000000000001
{'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'} | 0.1136
{'C': 1000, 'gamma': 0.0001, 'kernel': 'linear'} | 0.9032
{'C': 1000, 'gamma': 0.0001, 'kernel': 'poly'} | 0.9284000000000001
{'C': 1000, 'gamma': 0.0001, 'kernel': 'rbf'} | 0.11520000000000001

Where the optimized parameter values were:

```
SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

Accuracy on the test set with the optimized parameter values: 0.93
