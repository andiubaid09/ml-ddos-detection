import pandas as pd
import numpy as np
from google.colab import drive
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


drive.mount ('/content/drive')
datasheet = '/content/drive/My Drive/Datasheet/DDoS/dataset_sdn.csv'
df = pd.read_csv(datasheet)

# Cek Missing Values Kolom
print("Jumlah Missing Values :")
print(df.isnull().sum())

# Drop Missing Values
df = df.dropna(subset=['rx_kbps','tot_kbps'])
print(df.isnull().sum())
df.columns
Features = ['dt','dur','dur_nsec','tot_dur','pktrate','Protocol',
            'port_no', 'tx_kbps','rx_kbps','tot_kbps']
df_clean = df[Features]
numeric_feat = ['dt','dur','dur_nsec','tot_dur','pktrate','port_no','tx_kbps','rx_kbps','tot_kbps']
categorial_feat = ['Protocol']

# Membagi data dengan komposisi X = 80% dan y = 20% untuk model
X = df_clean
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42, stratify=y
)
# Menghitung rasio untuk imbalance
ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])
print("Ratio class imbalance (0/1)",ratio)

#Hasilnya akan menunjukkan Kelas 0(Normal) ada 1,56x lebih banyak daripada kelas 1(DDoS) di training set
scaler = StandardScaler()
num_features = numeric_feat
num_tranform = scaler

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder_cat = categorial_feat
encoder_tranform = encoder

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', num_tranform, num_features),
    ('cat', encoder_tranform, encoder_cat)
], remainder='drop')

# Membuat base Model LightGBM
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    objective='binary',
    metric ='auc',
    boosting_type ='gbdt',
    num_leaves = 31,
    learning_rate = 0.05,
    n_estimators= 1000,
    scale_pos_weight=ratio, #Uncomment jika ingin menangani imbalance
    random_state=42,
    n_jobs=-1
)

from sklearn.pipeline import Pipeline

lbgm_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', model)
])