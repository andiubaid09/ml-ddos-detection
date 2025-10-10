import pandas as pd
import numpy as np
from google.colab import drive
drive.mount ('/content/drive')

datasheet = '/content/drive/My Drive/Datasheet/DDoS/dataset_sdn.csv'
df = pd.read_csv(datasheet)