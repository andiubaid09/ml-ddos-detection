from xgboost import XGBClassifier

scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

xgb_clf = XGBClassifier(
    objective='binary:logistic',
    n_estimators= 200,
    learning_rate = 0.1,
    max_depth= 6,
    subsample = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = scale_pos_weight,
    random_state = 42,
    n_jobs= -1,
    eval_metrics= 'logloss'

)

from xgboost import XGBClassifier

XG_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', xgb_clf)
])
param_grid = {
    'classifier__max_depth' : [4,6,8],
    'classifier__learning_rate': [0.05, 0.1, 0.2],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__subsample': [0.7, 0.8, 1.0],
    'classifier__colsample_bytree': [0.7, 0.8, 1.0]
}

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    XG_pipeline,
    param_grid,
    cv = cv,
    scoring = 'accuracy',
    n_jobs = -1,
    verbose=2
)

# Jalankan GridSearchCV
random_search.fit(X_train, y_train)

print("Best Parameters", random_search.best_params_)
print("Best CV Score :", random_search.best_score_)