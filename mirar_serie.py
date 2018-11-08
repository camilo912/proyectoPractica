import numpy as np
import pandas as pd

df = pd.read_csv('data/datos_liquidez_nuevos.csv', header=0, index_col=0)
df = df['Merval']

import matplotlib.pyplot as plt

plt.plot(df.values)
plt.show()