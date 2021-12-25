from math import remainder
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

train = pd.read_csv("train-data.csv")
test = pd.read_csv("test-data.csv")

train = train.drop(["New_Price","Unnamed: 0"],axis=1)

train["Engine"]=train["Engine"].str.replace("CC","")
train["Power"]=train["Power"].str.replace("bhp","")
train["Mileage"]=train["Mileage"].str.replace("kmpl","")
train["Mileage"]=train["Mileage"].str.replace("km/kg","")

train["Engine"]=train["Engine"].astype(float)
train["Mileage"]=train["Mileage"].astype(float)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

X=train.drop(["Price"],axis=1)
y=train["Price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=61)

num_pipe1=make_pipeline(StandardScaler(),SimpleImputer(strategy="mean"))
cat_pipe2=make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder(handle_unknown="ignore"))
ct = make_column_transformer(
    (num_pipe1,["Year","Kilometers_Driven","Mileage","Engine","Seats"]),
    (cat_pipe2,["Name","Location","Fuel_Type","Transmission","Owner_Type","Power"]),
    remainder="passthrough")




## + name,location,year,km,fuel,transmission,ownertype,mileage,engine,power,seats)

x= ["Hyundai Creta 1.6 CRDi SX","Kochi",2015,32000,"Petrol"
                   ,"Manual","Second",26,19,1199,122]


from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=61)

R_pipeline=make_pipeline(ct,r_dt)
R_pipeline.fit(X_train,y_train)

print(R_pipeline.score(X_train,y_train))
print(R_pipeline.score(X_test,y_test))


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
reg =make_pipeline(ct,lr)
reg.fit(X_train,y_train)

print(reg.score(X_train,y_train))
print(reg.score(X_test,y_test))



print(R_pipeline.predict(R_pipeline.fit_transform([x])))





"""data = DataFrame(columns=["name","location","year","km","fuel","transmission","ownertype"
                          ,"millage","engine","power","seats"],index=[1])

data.xs(1)["name"]="Hyundai Creta 1.6 CRDi SX"
data.xs(1)["location"]="Kochi"
data.xs(1)["year"]=2015
data.xs(1)["km"]=32000
data.xs(1)["fuel"]="Petrol"
data.xs(1)["ownertype"]="Manual"
data.xs(1)["transmission"]="Second"
data.xs(1)["millage"]=26
data.xs(1)["engine"]=19
data.xs(1)["power"]=1199
data.xs(1)["seats"]=122

"""


#print(R_pipeline.predict(data))

"""x= ["Hyundai Creta 1.6 CRDi SX","Kochi",2015,32000,"Petrol"
                   ,"Manual","Second",26,19,1199,122]
print(x)
print(R_pipeline.predict(([x])))"""
