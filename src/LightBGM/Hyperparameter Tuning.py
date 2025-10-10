from scipy.stats import randint, uniform

param_distrib = {
    'classifier__n_estimators': randint(100, 600),
    'classifier__num_leaves': randint(20, 100),
    'classifier__max_depth': randint(3,15),
    'classifier__learning_rate': uniform(0.01,0.2),
    'classifier__subsample': uniform(0.6, 0.4),
    'classifier__colsample_bytree': uniform(0.6, 0.4),
    'classifier__reg_alpha': uniform(0,1),
    'classifier__reg_lambda':uniform(0,1)
}

from sklearn.model_selection import RandomizedSearchCV

random_searchlbgm = RandomizedSearchCV(
    estimator=lbgm_pipeline,
    param_distributions=param_distrib,
    n_iter = 15,
    cv = 5,
    scoring= 'accuracy',
    verbose= 2,
    random_state=42,
    n_jobs=-1


)
random_searchlbgm.fit(X_train, y_train)
print('Best params yang ditemukan :', random_searchlbgm.best_params_)
print('Best CV Score yang ditemukan :', random_searchlbgm.best_score_)
