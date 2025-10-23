import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('creditcard.csv')

# sc = StandardScaler()
# df['Amount'] = sc.fit_transform(pd.DataFrame(df['Amount']))
# df['Time'] = sc.fit_transform(pd.DataFrame(df['Time']))
da = df.drop_duplicates()

nor = da[da['Class'] == 0]
fra = da[da['Class'] == 1]

normal = nor.sample(n=2000)
fraud = fra.sample(n=2000, replace = True)

new = pd.concat([normal, fraud], ignore_index=True)

X = new.drop('Class', axis=1)
y = new['Class']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

model = RandomForestClassifier(random_state=42, n_estimators = 100)
model.fit(X_train,y_train)

# y_pred = model.predict(X_test)

# print(accuracy_score(y_pred,y_test)*100)

pickle.dump(model, open("model.pkl", "wb"))