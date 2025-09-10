#Model LightGBM
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    objective='binary',
    metric ='auc',
    boosting_type ='gbdt',
    num_leaves = 31,
    learning_rate = 0.05,
    n_estimators= 1000,
    #scale_pos_weight=ratio, #Uncomment jika ingin menangani imbalance
    random_state=42
)
#Training Model
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(100)]
)
