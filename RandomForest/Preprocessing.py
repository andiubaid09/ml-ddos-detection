import pandas as pd
import numpy as np
from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/My Drive/Skripsi/Datasheet/DH_label_only.csv'

df_label = pd.read_csv(file_path)
print("Jumlah Missing Values:")
print(df_label.isnull().sum())
df_label = df_label.dropna(subset=['tx_kbps', 'tot_kbps'])
df_label['label'] = df_label['label'].astype(int)

df_label.info()

from sklearn.preprocessing import OneHotEncoder

df_label['Protocol'] = df_label['Protocol'].astype(str)

encoder = OneHotEncoder(sparse_output=False)
encoded_clean = encoder.fit_transform(df_label[['Protocol']]).astype(int)
encoded_clean = pd.DataFrame(
    encoded_clean,
    columns=encoder.get_feature_names_out(['Protocol']),
    index=df_label.index #Baris tetap sinkron
)
df_label = pd.concat([df_label.drop(columns=['Protocol']), encoded_clean], axis=1)
df_label.info()
