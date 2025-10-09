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
