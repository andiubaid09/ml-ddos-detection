from google.colab import drive
import pandas as pd

drive.mount('/content/drive')
datasheet = '/content/drive/My Drive/Datasheet/DDoS/dataset_sdn.csv'
df = pd.read_csv(datasheet)