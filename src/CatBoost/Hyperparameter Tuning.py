from scipy.stats import randint, uniform

param_distri = {
    # Kedalaman Pohon
    'classifier__depth':randint(4,11),
    # Learning Rate
    'classifier__learning_rate': uniform(0.01, 0.29),
    # L2 regularization di antara 1-10
    'classifier__l2_leaf_reg': randint(1,10),
    # Jumlah estimator pohon
    'classifier__n_estimators': randint(100,400),
    # Subsample rate
    'classifier__subsample': uniform(0.5,0.5),
}

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator = cat_pipeline,
    param_distributions = param_distri,
    n_iter = 15,
    cv = 5,
    scoring = 'accuracy',
    verbose = 2,
    random_state=42,
    n_jobs=-1

)
# Memberitahu model CatBoost kolom kategorial dari pipeline [fitur hasil scaling numerik] + [fitur yang dipassthrough]
cat_feature_indices = [len(Numeric_features)]  # kalau cuma 1 kolom kategori
random_search.fit(X_train, y_train, classifier__cat_features=cat_feature_indices)
print("Best params :", random_search.best_params_)
print("Best CV Score: ", random_search.best_score_)
