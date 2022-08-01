#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra -for numeric computations
import pandas as pd # data processing -to store data as dataframes 
import matplotlib.pyplot as plt # data visualization 
import seaborn as sns # data visualization 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math


# In[ ]:


data = pd.read_csv("https://bit.ly/prosper-dataset")
data.head(7)


# In[ ]:


# parsing Dates
data['ListingCreationDate'] = pd.to_datetime(data['ListingCreationDate'])
data['ClosedDate'] = pd.to_datetime(data['ClosedDate'])
data['DateCreditPulled'] = pd.to_datetime(data['DateCreditPulled'])
data['FirstRecordedCreditLine'] = pd.to_datetime(data['FirstRecordedCreditLine'])
data['LoanOriginationDate'] = pd.to_datetime(data['LoanOriginationDate'])


# In[ ]:


#  removing any feature with more than 75% of missing values.
data_with_less_missing_values = data.dropna(thresh=data.shape[0] * 0.25, axis=1)
data_with_less_missing_values.shape


# In[ ]:


# removing loan samples with have more than 20% of missing values
data_with_less_missing_values = data_with_less_missing_values.dropna(thresh=data.shape[1] * 0.80, axis=0).reset_index(drop=True)
data_with_less_missing_values.shape


# In[ ]:


cat_cols = [name for name in data_with_less_missing_values 
                        if data_with_less_missing_values[name].dtype in ["object", "bool" ]]
numerical_cols = [name for name in data_with_less_missing_values.columns
                      if data_with_less_missing_values[name].dtype in ['int64', 'float64', 'datetime64[ns]']]
cat_data = data_with_less_missing_values.drop(axis=1, columns=numerical_cols)
num_data = data_with_less_missing_values.drop(axis=1, columns=cat_cols)


# In[ ]:


cat_data.Occupation = cat_data.Occupation.fillna(cat_data.Occupation.mode().iloc[0])


# In[ ]:


cat_data = cat_data.drop(axis=1, columns=['ProsperRating (Alpha)'])


# In[ ]:


# all missing values in the CreditGrade column represents the rating value 0
# fill in nan values with letter Z and then use OrdinalEncoder to convert it to numerical values
cat_data.CreditGrade = cat_data.CreditGrade.fillna("Z")
from sklearn.preprocessing import OrdinalEncoder
ratings = ['Z', 'HR', 'E', 'D', 'C', 'B', 'A', 'AA']
encoder = OrdinalEncoder(categories = [ratings])
cat_data[['CreditGrade']] = encoder.fit_transform(cat_data[['CreditGrade']])
cat_data.CreditGrade = cat_data.CreditGrade.astype(int)


# In[ ]:





# In[ ]:


cat_data = cat_data.drop(columns=['LoanKey','MemberKey','ListingKey'],axis=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


num_data['DebtToIncomeRatio'].fillna(value=num_data['DebtToIncomeRatio'].median(), inplace=True)


# In[ ]:


num_data.EmploymentStatusDuration = num_data.EmploymentStatusDuration.fillna(num_data.EmploymentStatusDuration.mode().iloc[0])


# In[ ]:


num_data.ClosedDate.fillna(value='Not Closed', inplace=True)


# In[ ]:


num_data.isna().sum()


# # added new

# In[ ]:


num_data['EstimatedEffectiveYield'].fillna(value=num_data['EstimatedEffectiveYield'].median(), inplace=True)
num_data['EstimatedLoss'].fillna(value=num_data['EstimatedLoss'].median(), inplace=True)
num_data['EstimatedReturn'].fillna(value=num_data['EstimatedReturn'].median(), inplace=True)
num_data['ProsperRating (numeric)'].fillna(value=num_data['ProsperRating (numeric)'].median(), inplace=True)
num_data['ProsperScore'].fillna(value=num_data['ProsperScore'].median(), inplace=True)


# In[ ]:


num_data.isna().sum()


# In[ ]:


num_data.select_dtypes(include=('object'))


# In[ ]:


num_data['ClosedDate'] = num_data['ClosedDate'].fillna(0)
num_data['ClosedDate'] = num_data['ClosedDate'].apply(lambda x:1 if x!='Not Closed' else 0)


# In[ ]:


num_data.ClosedDate


# In[ ]:


modified_data = num_data.join(cat_data)


# In[ ]:


modified_data.select_dtypes(include=('bool')).columns


# In[ ]:


bools = ['IsBorrowerHomeowner', 'CurrentlyInGroup', 'IncomeVerifiable']
for i in bools:
    modified_data[i] = modified_data[i].apply(lambda x:1 if x else 0)


# In[ ]:


modified_data.info()


# In[ ]:


# modified_data.to_csv('cleaned.csv',index=True)


# In[ ]:


modified_data = modified_data.drop(["ListingCreationDate","DateCreditPulled","FirstRecordedCreditLine","LoanOriginationDate","LoanOriginationQuarter","ListingNumber","LoanNumber"],axis=1)


# In[ ]:


# Selected those features according to the output of RandomForestClassifier importance function
modified_data = modified_data[["ClosedDate","LoanCurrentDaysDelinquent","LoanMonthsSinceOrigination","LP_CustomerPrincipalPayments","LP_GrossPrincipalLoss","LP_NetPrincipalLoss","LP_CustomerPayments","EmploymentStatus","LP_ServiceFees","LoanOriginalAmount","Investors","EstimatedReturn","LP_InterestandFees","MonthlyLoanPayment","LP_CollectionFees","EstimatedEffectiveYield","EstimatedLoss","Term","BorrowerAPR","LP_NonPrincipalRecoverypayments","BorrowerRate","ListingCategory (numeric)","LenderYield","CreditScoreRangeUpper","OpenRevolvingMonthlyPayment","ProsperScore","CreditScoreRangeLower","RevolvingCreditBalance","ProsperRating (numeric)","AvailableBankcardCredit","EmploymentStatusDuration","DebtToIncomeRatio","StatedMonthlyIncome","BankcardUtilization","TotalCreditLinespast7years","TotalTrades","LoanStatus"]]

y = modified_data["LoanStatus"]
X = modified_data.drop(["LoanStatus"],axis=1)
label_encoding_cols=["EmploymentStatus"]
for i in label_encoding_cols:
    X[i]=X[i].astype("category")
    X[i]=X[i].cat.codes


# In[ ]:





# In[ ]:


x_train, x_test,y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
rf = RandomForestClassifier(n_estimators = 300,random_state = 1, max_depth=30,n_jobs=-1)
rf.fit(x_train,y_train)
rf_pred=rf.predict(x_test)

import pickle 
pickle.dump(rf,open('rf_classification.sav','wb'))



# In[ ]:


cols = ['ClosedDate', 'LoanCurrentDaysDelinquent', 'LoanMonthsSinceOrigination',
       'LP_CustomerPrincipalPayments', 'LP_GrossPrincipalLoss',
       'LP_NetPrincipalLoss', 'LP_CustomerPayments', 'EmploymentStatus',
       'LP_ServiceFees', 'LoanOriginalAmount', 'Investors', 'EstimatedReturn',
       'LP_InterestandFees', 'MonthlyLoanPayment', 'LP_CollectionFees',
       'EstimatedEffectiveYield', 'EstimatedLoss', 'Term', 'BorrowerAPR',
       'LP_NonPrincipalRecoverypayments', 'BorrowerRate',
       'ListingCategory (numeric)', 'LenderYield', 'CreditScoreRangeUpper',
       'OpenRevolvingMonthlyPayment', 'ProsperScore', 'CreditScoreRangeLower',
       'RevolvingCreditBalance', 'ProsperRating (numeric)',
       'AvailableBankcardCredit', 'EmploymentStatusDuration',
       'DebtToIncomeRatio', 'StatedMonthlyIncome', 'BankcardUtilization',
       'TotalCreditLinespast7years', 'TotalTrades']
for i in cols:
    print(f'{i} = models.FloatField(default=0)')


# In[ ]:


x_train


# In[ ]:


print("---------------------------------------------------------------------")
print("Accuracy Score for Random Forest :",accuracy_score(y_test,rf_pred))
print("---------------------------------------------------------------------")
print("\n")
print("classification stats for Random Forest Classifier :\n\n",classification_report(y_test, rf_pred))
print("---------------------------------------------------------------------")


# In[ ]:


cm=confusion_matrix(y_test,rf_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:1','Predicted:2','Predicted:3','Predicted:4','Predicted:5','Predicted:6','Predicted:7','Predicted:8','Predicted:9','Predicted:10','Predicted:11'],
                                         index=['Actual:1','Actual:2','Actual:3','Actual:4','Actual:5','Actual:6','Actual:7','Actual:8','Actual:9','Actual:10','Actual:11'])                          
plt.figure(figsize = (15,15))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap='Blues')
plt.title("confusion Matrix for  Random Forest")
plt.show()


# In[ ]:


y = modified_data["BorrowerRate"]
X = modified_data.drop(["BorrowerRate"],axis=1)
label_encoding_cols=["EmploymentStatus","LoanStatus"]
for i in label_encoding_cols:
    X[i]=X[i].astype("category")
    X[i]=X[i].cat.codes


# In[ ]:


x_train, x_test,y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
rf = RandomForestRegressor(n_estimators = 300,random_state = 1, max_depth=30,n_jobs=-1)
rf.fit(x_train,y_train)
rf_pred=rf.predict(x_test)


# In[ ]:


import pickle
pickle.dump(rf,open('rf_regression.sav','wb'))


# In[ ]:


RMSE = math.sqrt(mean_squared_error(y_test,rf_pred))
print('RMSE:'+str(RMSE))
MAE = mean_absolute_error(y_test,rf_pred)
print('MAE:'+str(MAE))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




