# Results for the SVM assignement
For a cross-validation done by ```GridSearchCV``` from ```sklearn```, the following accuracies were got:

```params``` | mean accuracy (```mean_test_score```)
--- | ---
{'C': 0.001, 'kernel': 'linear'} | 0.9
{'C': 1000, 'kernel': 'linear'} | 0.9
{'C': 10000000, 'kernel': 'linear'} | 0.9

Where the optimized parameter values were:

```
SVC(C=0.001, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

Accuracy on the test set with the optimized parameter values: 1.0
