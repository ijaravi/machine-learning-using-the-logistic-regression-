#importing required libraries 
import numpy as np
import pandas as pd 
import pylab as pl
import statsmodels.api as sm


#insertin thw dataset 
df=pd.read_csv("binary.csv")
df.head()
# here we do logistic regression for data present 
# where as in thee linear wee will be creating it  newly

#rename.column names 
df.columns=["admit","gre","gpa","prestige"]
df.head() 

df.shape

df.describe()

# frequency  according to prestige column 
pd.crosstab(df['admit'],df['prestige'],rownames=['admit']) # here we can se the dependencies of the two columns

df.hist()

#creating the dummies
rankdummy=pd.get_dummies(df['prestige'],prefix='prestige')
rankdummy.head()

cols_to_keep=['admit','gre','gpa']
data=df[cols_to_keep].join(rankdummy.loc[:,'prestige_2':])   
data.head()

data['intercept']=1.0
data.head()

#performing regression 
train_cols=data.columns[1:]
    
logit=sm.Logit(data['admit'],data[train_cols])
result=logit.fit()

abc=result.predict([660,3.67,0,1,0,1.0])
print(abc)

result.summary()