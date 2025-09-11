# Membuat model XGBoost
import xgboost as xgb
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=500,
    learning_rate = 0.1,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train.value_counts()[0]/ y_train.value_counts()[1]),
    eval_metrics='logloss',
    use_label_encoder=False
)
#Training Model
model.fit(
    X_train, y_train,
)
