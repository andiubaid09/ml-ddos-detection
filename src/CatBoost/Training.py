# Membuat model CatBoos
from catboost import CatBoostClassifier, Pool

modelCat = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.1,
    loss_function='Logloss',
    eval_metric='F1',
    cat_features=categorical_cols,
    random_seed=42,
    verbose=100,
)
modelCat.fit(X_train, y_train, eval_set=(X_test, y_test))
