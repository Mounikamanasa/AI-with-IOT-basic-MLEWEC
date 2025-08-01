import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

df=pd.read_csv('feeds (3).csv')


label_encoder=LabelEncoder()
df['action']=label_encoder.fit_transform(df['action'])

X=df[['temperature','humidity']]
Y=df['action']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(X_train,Y_train)

joblib.dump(model,'dc_model.pkl')

print("model trained and downloaded successfully")
