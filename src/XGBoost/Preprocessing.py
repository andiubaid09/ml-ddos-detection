import pandas as pd
import numpy as np
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

drive.mount ('/content/drive')

datasheet = '/content/drive/My Drive/Datasheet/dataset_sdn.csv'
df = pd.read_csv(datasheet)
Features = [
    'dt','dur','dur_nsec','tot_dur','pktrate','Protocol','port_no','tx_kbps','rx_kbps','tot_kbps'
]
df_clean = df[Features]

X = df_clean
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder_cat = ['Protocol']
encoder_transform = encoder
numeric_features = ['dt','dur','dur_nsec','tot_dur','pktrate','port_no','tx_kbps','rx_kbps','tot_kbps']
scaler = StandardScaler()
numeric_feat = numeric_features
numeric_transform = scaler

preprocessor = ColumnTransformer(
    transformers=[
        ('cat',encoder_transform, encoder_cat),
        ('num', numeric_transform, numeric_feat)
    ],
    remainder = 'drop'
)