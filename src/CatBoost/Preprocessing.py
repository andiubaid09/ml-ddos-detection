import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

datasheet = '/content/drive/My Drive/Datasheet/DDoS/dataset_sdn.csv'
df = pd.read_csv(datasheet)
Features = [
    'dt','dur','dur_nsec','tot_dur','pktrate','Protocol','port_no','tx_kbps',
    'rx_kbps','tot_kbps'
]
df_clean = df[Features]

from sklearn.model_selection import train_test_split

X = df_clean
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42, stratify=y
)
Numeric_features = ['dt','dur','dur_nsec','tot_dur','pktrate','port_no','tx_kbps','rx_kbps','tot_kbps']
Kategorial_features = ['Protocol']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
num_features = Numeric_features
num_transform = scaler

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', num_transform, num_features)
], remainder="passthrough")

# Membuat base model CatBoos
from catboost import CatBoostClassifier, Pool

modelCat = CatBoostClassifier(
    loss_function='Logloss',
    depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=0,
    eval_metric='Accuracy'
)

from sklearn.pipeline import Pipeline

cat_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier',modelCat)
])