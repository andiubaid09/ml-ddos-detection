import pandas as pd
import numpy as np
from google.colab import drive
drive.mount ('/content/drive')

datasheet = '/content/drive/My Drive/Datasheet/dataset_sdn.csv'
df = pd.read_csv(datasheet)
df.info()

from sklearn.preprocessing import OneHotEncoder

df['Protocol'] = df ['Protocol'].astype(str)
encoder = OneHotEncoder(sparse_output=False)
encoded_clean = encoder.fit_transform(df[['Protocol']]).astype(str)
encoded_clean = pd.DataFrame(
    encoded_clean,
    columns= encoder.get_feature_names_out(['Protocol']),
    index = df.index
)
df = pd.concat([df.drop(columns=['Protocol']), encoded_clean], axis = 1)
df.columns
Features = [
    'dt','dur','dur_nsec','tot_dur','pktrate','Protocol_ICMP','Protocol_TCP','Protocol_UDP',
    'port_no','tx_kbps','rx_kbps','tot_kbps'
]
df_clean = df[Features]

from sklearn.model_selection import train_test_split

X = df_clean.astype('float32')
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state=42, stratify =y
)
df['label'].value_counts()
