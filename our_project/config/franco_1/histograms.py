import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


thepath = 'our_project/config/franco/one_digit_errors.txt'
with open(thepath, 'r') as f:
    lines = f.readlines()

n = 500
solutions = [line[-5:-3] for line in lines[-n:]]
predictions = [line[-11:-9] for line in lines[-n:]]
inputs = [line[1:4] for line in lines[-n:]]

plt.figure()
pd.Series(solutions).value_counts().plot(kind='bar')
plt.show()
plt.figure()
pd.Series(predictions).value_counts().plot(kind='bar')
plt.show()
plt.figure()
pd.Series(inputs).value_counts().plot(kind='bar')
plt.show()





