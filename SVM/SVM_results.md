# Results for the SVM assignement
For a cross-validation done by ```GridSearchCV``` from ```sklearn```, the following accuracies were got:

```params``` | mean accuracy (```mean_test_score```)
--- | ---
{'C': 0.001, 'gamma': 0.1, 'kernel': 'linear'} | 0.9153
{'C': 0.001, 'gamma': 0.1, 'kernel': 'poly'} | 0.9494
{'C': 0.001, 'gamma': 0.1, 'kernel': 'rbf'} | 0.1135
{'C': 0.001, 'gamma': 0.01, 'kernel': 'linear'} | 0.9153
{'C': 0.001, 'gamma': 0.01, 'kernel': 'poly'} | 0.9494
{'C': 0.001, 'gamma': 0.01, 'kernel': 'rbf'} | 0.1135
{'C': 0.001, 'gamma': 0.0001, 'kernel': 'linear'} | 0.9153
{'C': 0.001, 'gamma': 0.0001, 'kernel': 'poly'} | 0.9494
{'C': 0.001, 'gamma': 0.0001, 'kernel': 'rbf'} | 0.1135
{'C': 0.001, 'gamma': 1e-06, 'kernel': 'linear'} | 0.9153
{'C': 0.001, 'gamma': 1e-06, 'kernel': 'poly'} | 0.9007999999999999
{'C': 0.001, 'gamma': 1e-06, 'kernel': 'rbf'} | 0.1135
{'C': 0.001, 'gamma': 1e-07, 'kernel': 'linear'} | 0.9153
{'C': 0.001, 'gamma': 1e-07, 'kernel': 'poly'} | 0.1135
{'C': 0.001, 'gamma': 1e-07, 'kernel': 'rbf'} | 0.1135
{'C': 0.001, 'gamma': 1e-08, 'kernel': 'linear'} | 0.9153
{'C': 0.001, 'gamma': 1e-08, 'kernel': 'poly'} | 0.1135
{'C': 0.001, 'gamma': 1e-08, 'kernel': 'rbf'} | 0.1135
{'C': 0.1, 'gamma': 0.1, 'kernel': 'linear'} | 0.9153
{'C': 0.1, 'gamma': 0.1, 'kernel': 'poly'} | 0.9494
{'C': 0.1, 'gamma': 0.1, 'kernel': 'rbf'} | 0.1135
{'C': 0.1, 'gamma': 0.01, 'kernel': 'linear'} | 0.9153
{'C': 0.1, 'gamma': 0.01, 'kernel': 'poly'} | 0.9494
{'C': 0.1, 'gamma': 0.01, 'kernel': 'rbf'} | 0.1135
{'C': 0.1, 'gamma': 0.0001, 'kernel': 'linear'} | 0.9153
{'C': 0.1, 'gamma': 0.0001, 'kernel': 'poly'} | 0.9494
{'C': 0.1, 'gamma': 0.0001, 'kernel': 'rbf'} | 0.1135
{'C': 0.1, 'gamma': 1e-06, 'kernel': 'linear'} | 0.9153
{'C': 0.1, 'gamma': 1e-06, 'kernel': 'poly'} | 0.9504999999999999
{'C': 0.1, 'gamma': 1e-06, 'kernel': 'rbf'} | 0.6777
{'C': 0.1, 'gamma': 1e-07, 'kernel': 'linear'} | 0.9153
{'C': 0.1, 'gamma': 1e-07, 'kernel': 'poly'} | 0.7070000000000001
{'C': 0.1, 'gamma': 1e-07, 'kernel': 'rbf'} | 0.9097
{'C': 0.1, 'gamma': 1e-08, 'kernel': 'linear'} | 0.9153
{'C': 0.1, 'gamma': 1e-08, 'kernel': 'poly'} | 0.1135
{'C': 0.1, 'gamma': 1e-08, 'kernel': 'rbf'} | 0.7877
{'C': 1, 'gamma': 0.1, 'kernel': 'linear'} | 0.9153
{'C': 1, 'gamma': 0.1, 'kernel': 'poly'} | 0.9494
{'C': 1, 'gamma': 0.1, 'kernel': 'rbf'} | 0.1135
{'C': 1, 'gamma': 0.01, 'kernel': 'linear'} | 0.9153
{'C': 1, 'gamma': 0.01, 'kernel': 'poly'} | 0.9494
{'C': 1, 'gamma': 0.01, 'kernel': 'rbf'} | 0.1135
{'C': 1, 'gamma': 0.0001, 'kernel': 'linear'} | 0.9153
{'C': 1, 'gamma': 0.0001, 'kernel': 'poly'} | 0.9494
{'C': 1, 'gamma': 0.0001, 'kernel': 'rbf'} | 0.1135
{'C': 1, 'gamma': 1e-06, 'kernel': 'linear'} | 0.9153
{'C': 1, 'gamma': 1e-06, 'kernel': 'poly'} | 0.9494
{'C': 1, 'gamma': 1e-06, 'kernel': 'rbf'} | 0.9513
{'C': 1, 'gamma': 1e-07, 'kernel': 'linear'} | 0.9153
{'C': 1, 'gamma': 1e-07, 'kernel': 'poly'} | 0.9007999999999999
{'C': 1, 'gamma': 1e-07, 'kernel': 'rbf'} | 0.945
{'C': 1, 'gamma': 1e-08, 'kernel': 'linear'} | 0.9153
{'C': 1, 'gamma': 1e-08, 'kernel': 'poly'} | 0.1135
{'C': 1, 'gamma': 1e-08, 'kernel': 'rbf'} | 0.9048
{'C': 10, 'gamma': 0.1, 'kernel': 'linear'} | 0.9153
{'C': 10, 'gamma': 0.1, 'kernel': 'poly'} | 0.9494
{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} | 0.1135
{'C': 10, 'gamma': 0.01, 'kernel': 'linear'} | 0.9153
{'C': 10, 'gamma': 0.01, 'kernel': 'poly'} | 0.9494
{'C': 10, 'gamma': 0.01, 'kernel': 'rbf'} | 0.1135
{'C': 10, 'gamma': 0.0001, 'kernel': 'linear'} | 0.9153
{'C': 10, 'gamma': 0.0001, 'kernel': 'poly'} | 0.9494
{'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'} | 0.1135
{'C': 10, 'gamma': 1e-06, 'kernel': 'linear'} | 0.9153
{'C': 10, 'gamma': 1e-06, 'kernel': 'poly'} | 0.9494
{'C': 10, 'gamma': 1e-06, 'kernel': 'rbf'} | 0.9522999999999999
{'C': 10, 'gamma': 1e-07, 'kernel': 'linear'} | 0.9153
{'C': 10, 'gamma': 1e-07, 'kernel': 'poly'} | 0.9477
{'C': 10, 'gamma': 1e-07, 'kernel': 'rbf'} | 0.9579000000000001
{'C': 10, 'gamma': 1e-08, 'kernel': 'linear'} | 0.9153
{'C': 10, 'gamma': 1e-08, 'kernel': 'poly'} | 0.15660000000000002
{'C': 10, 'gamma': 1e-08, 'kernel': 'rbf'} | 0.9304
{'C': 1000, 'gamma': 0.1, 'kernel': 'linear'} | 0.9153
{'C': 1000, 'gamma': 0.1, 'kernel': 'poly'} | 0.9494
{'C': 1000, 'gamma': 0.1, 'kernel': 'rbf'} | 0.1135
{'C': 1000, 'gamma': 0.01, 'kernel': 'linear'} | 0.9153
{'C': 1000, 'gamma': 0.01, 'kernel': 'poly'} | 0.9494
{'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'} | 0.1135
{'C': 1000, 'gamma': 0.0001, 'kernel': 'linear'} | 0.9153
{'C': 1000, 'gamma': 0.0001, 'kernel': 'poly'} | 0.9494
{'C': 1000, 'gamma': 0.0001, 'kernel': 'rbf'} | 0.1135
{'C': 1000, 'gamma': 1e-06, 'kernel': 'linear'} | 0.9153
{'C': 1000, 'gamma': 1e-06, 'kernel': 'poly'} | 0.9494
{'C': 1000, 'gamma': 1e-06, 'kernel': 'rbf'} | 0.9522999999999999
{'C': 1000, 'gamma': 1e-07, 'kernel': 'linear'} | 0.9153
{'C': 1000, 'gamma': 1e-07, 'kernel': 'poly'} | 0.9494
{'C': 1000, 'gamma': 1e-07, 'kernel': 'rbf'} | 0.9575999999999999
{'C': 1000, 'gamma': 1e-08, 'kernel': 'linear'} | 0.9153
{'C': 1000, 'gamma': 1e-08, 'kernel': 'poly'} | 0.9007999999999999
{'C': 1000, 'gamma': 1e-08, 'kernel': 'rbf'} | 0.9333

Where the optimized parameter values were:

```
SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1e-07, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

Accuracy on the test set with the optimized parameter values: 0.9995
