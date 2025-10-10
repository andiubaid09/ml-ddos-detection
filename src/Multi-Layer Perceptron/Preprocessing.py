import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

datasheet = '/content/drive/My Drive/Datasheet/DDoS/dataset_sdn.csv'

df = pd.read_csv(datasheet)
print("Jumlah Missing Values")
print(df.isnull().sum())
df = df.dropna(subset=['rx_kbps','tot_kbps'])
df['label'] = df['label'].astype(int)

df.info()
Features = [
    'dt','dur','dur_nsec','tot_dur','pktrate','Protocol','port_no','tx_kbps','rx_kbps','tot_kbps'
]
df_fitur = df[Features]

from sklearn.model_selection import train_test_split

X = df_fitur
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,y_train, test_size = 0.2, random_state=42, stratify=y_train
)
numeric_features = ['dt','dur','dur_nsec','tot_dur','pktrate','port_no','tx_kbps','rx_kbps','tot_kbps']
categorial_features = ['Protocol']

from sklearn.preprocessing import StandardScaler, OneHotEncoder

scaler = StandardScaler()
num_feat = numeric_features
num_tranform = scaler
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_feat = categorial_features
cat_transform = encoder

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transform, cat_feat),
        ('num', num_tranform, num_feat)
    ],
    remainder = 'drop'
)
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.fit_transform(X_test)
X_val_prep = preprocessor.fit_transform(X_val)
input_dim = X_train_prep.shape[1]