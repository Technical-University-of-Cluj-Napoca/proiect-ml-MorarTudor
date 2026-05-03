import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

# 1. Date pentru Clasificare (Aprobarea unui credit bancar)
X_cls, y_cls = make_classification(n_samples=1800, n_features=10, n_informative=7, random_state=42)
df_cls = pd.DataFrame(X_cls, columns=[f'Feature_{i}' for i in range(1, 11)])
df_cls['Aprobare_Credit'] = y_cls
df_cls.to_csv('data/clasificare.csv', index=False)
print("Setul de date pentru clasificare a fost generat cu succes!")

# 2. Date pentru Regresie (Predicția prețului la motorină)
X_reg, y_reg = make_regression(n_samples=1800, n_features=8, noise=0.1, random_state=42)
df_reg = pd.DataFrame(X_reg, columns=[f'Feature_{i}' for i in range(1, 9)])
y_reg_scaled = np.interp(y_reg, (y_reg.min(), y_reg.max()), (6.0, 8.5))
df_reg['Pret_Motorina'] = np.round(y_reg_scaled, 2)
df_reg.to_csv('data/regresie.csv', index=False)
print("Setul de date pentru regresie a fost generat cu succes!")