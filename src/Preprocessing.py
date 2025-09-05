import pandas as pd
df = pd.read_csv(datasheet)

#Mengecek info DataFrame
print('Info dari Datasheet\n')
df.info()

#Mengecek ukuran dataFrame (jumlah baris, jumlah kolom)
print('Jumlah baris dan Jumlah kolom datasheet ini:')
df.shape

#Distribusi dari kolom label
df['label'].value_counts()
print('Jumlah Missing Values')
print(df.isnull().sum())

# Menghapus kolom bernilai null
df = df.dropna(subset=['rx_kbps','tot_kbps'])
df.info()
from sklearn.preprocessing import OneHotEncoder

df['Protocol'] = df['Protocol'].astype(str)

encoding = OneHotEncoder(sparse_output=False)
encoded  = encoding.fit_transform(df[['Protocol']]).astype(int)
encoded  = pd.DataFrame(
    encoded,
    columns=encoding.get_feature_names_out(['Protocol']),
    index= df.index
)
df = pd.concat([df.drop(columns=['Protocol']), encoded], axis =1)
df.columns
