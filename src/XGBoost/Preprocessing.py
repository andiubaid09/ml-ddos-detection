import pandas as pd
import numpy as np
from google.colab import drive
drive.mount ('/content/drive')

datasheet = '/content/drive/My Drive/Datasheet/dataset_sdn.csv'
df = pd.read_csv(datasheet)
Features = [
    'dt','dur','dur_nsec','tot_dur','pktrate','Protocol','port_no','tx_kbps','rx_kbps','tot_kbps'
]
df_clean = df[Features]

from sklearn.model_selection import train_test_split

X = df_clean
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)

from sklearn.preprocessing import OneHotEncoder, StandardScaler

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder_cat = ['Protocol']
encoder_transform = encoder
numeric_features = ['dt','dur','dur_nsec','tot_dur','pktrate','port_no','tx_kbps','rx_kbps','tot_kbps']
scaler = StandardScaler()
numeric_feat = numeric_features
numeric_transform = scaler

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('cat',encoder_transform, encoder_cat),
        ('num', numeric_transform, numeric_feat)
    ],
    remainder = 'drop'
)
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

from xgboost import XGBClassifier

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