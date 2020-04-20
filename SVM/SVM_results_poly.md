# Results for the SVM assignement
For a cross-validation done by ```GridSearchCV``` from ```sklearn```, the following accuracies were got:

```params``` | mean accuracy (```mean_test_score```)
--- | ---
{'C': 0.001, 'degree': 2, 'gamma': 0.1, 'kernel': 'poly'} | 0.9317499999999999
{'C': 0.001, 'degree': 2, 'gamma': 0.01, 'kernel': 'poly'} | 0.9317499999999999
{'C': 0.001, 'degree': 3, 'gamma': 0.1, 'kernel': 'poly'} | 0.9229999999999998
{'C': 0.001, 'degree': 3, 'gamma': 0.01, 'kernel': 'poly'} | 0.9229999999999998
{'C': 1000, 'degree': 2, 'gamma': 0.1, 'kernel': 'poly'} | 0.9317499999999999
{'C': 1000, 'degree': 2, 'gamma': 0.01, 'kernel': 'poly'} | 0.9317499999999999
{'C': 1000, 'degree': 3, 'gamma': 0.1, 'kernel': 'poly'} | 0.9229999999999998
{'C': 1000, 'degree': 3, 'gamma': 0.01, 'kernel': 'poly'} | 0.9229999999999998

Where the optimized parameter values were:

```
SVC(C=0.001, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=2, gamma=0.1, kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

Accuracy on the test set with the optimized parameter values: 1.0
