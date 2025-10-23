import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('creditcard.csv')
da = df.drop_duplicates()
nor = da[da['Class'] == 0]
fra = da[da['Class'] == 1]
normal = nor.sample(n=2000, random_state=42)
fraud = fra.sample(n=2000, replace = True, random_state=42)
new = pd.concat([normal, fraud], ignore_index=True)
X = new.drop('Class', axis=1)
y = new['Class']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
model = RandomForestClassifier(random_state=42, n_estimators = 100, n_jobs = -1)
model.fit(X_train,y_train)

joblib.dump(model, "model.joblib")
# pickle.dump(model, open("model.pkl", "wb"))