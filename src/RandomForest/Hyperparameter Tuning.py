from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


param_distributions = {
    'classifier__n_estimators': randint(100,200,300),
    'classifier__max_depth': randint(4, 5, 10),
    'classifier__min_samples_split': randint(2, 5, 10),
    'classifier__min_samples_leaf': randint(1, 3, 10),
    'classifier__max_features': ['sqrt', 'log2', None],
    'classifier__bootstrap': [True,False]
}
cv = StratifiedKFold(n_splits=5 , shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_distributions,
    n_iter = 30,
    cv = cv,
    scoring = 'accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)
random_search.fit(X_train, y_train)
print('Best parameters', random_search.best_params_)
print('Best CV Score', random_search.best_score_)