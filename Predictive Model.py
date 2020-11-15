#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import datetime as dt


# In[2]:


#reading ths csv file
df_covid = pd.read_csv("COVID-19_US-counties.csv", parse_dates=['date'], dayfirst=True)
print(df_covid)


# In[3]:


#information of the file
df_covid.info()


# In[4]:


#descriotion of the data
df_covid.describe()


# In[5]:


#taking the most recent information - October 18 2020 
oct18 = df_covid[df_covid.date =='2020-10-18']


# In[6]:


print(oct18)


# In[7]:


#removing nan values
oct18 = oct18.dropna()
print("COVID-19 Cases & Death per county by State for 18th October is as follows ")
print(oct18)


# In[8]:


#reading ths GDP XLSX file
df_gdp = pd.read_excel("GDP_Counties.xlsx")
print(df_gdp)


# In[9]:


#removing nan values
df_gdp = df_gdp.dropna()
print("Gdp by Counties are as follows ")
print(df_gdp)


# In[10]:


#merging GDP & COVID cases file
df_covid_gdp = pd.merge(oct18,df_gdp,on=["county", "state"])
print(df_covid_gdp)


# In[ ]:





# In[11]:


#reading ths housing xlsx file
df_housing = pd.read_excel("CO-EST2019-ANNHU.xlsx")
print(df_housing)


# In[ ]:





# In[12]:


#merging Housing, GDP, and COVID cases file
df_covid_gdp_housing = pd.merge(df_covid_gdp,df_housing,on=["county","state"])
print(df_covid_gdp_housing)


# In[ ]:





# In[13]:


#reading the census XLSX file
df_census = pd.read_excel("census total.xlsx")
print(df_census)


# In[14]:


#merging Census, Housing, GDP, and COVID cases file
df_covid_gdp_housing_census = pd.merge(df_covid_gdp_housing,df_census,on=["county","fips"])
print(df_covid_gdp_housing_census)


# In[15]:


#renaming the file as df
df = df_covid_gdp_housing_census
print(df)


# In[16]:


#dropping fips code
df.drop('fips',axis=1,inplace=True)


# In[17]:


#creating target variable "cases/100 pop" and "deaths/100 po"
df_covid_gdp_housing_census['cases/100 population'] = ((df_covid_gdp_housing_census['cases']/df_covid_gdp_housing_census['Total Population'])*100)
df_covid_gdp_housing_census['deaths/100 population'] = ((df_covid_gdp_housing_census['deaths']/df_covid_gdp_housing_census['Total Population'])*100)
df.describe()


# In[18]:


#creating a data frame to check top counties for cases
max_confirmed_cases_bycounties = df.sort_values(by="cases", ascending=False)
max_confirmed_cases_bycounties.head(10)
top_counties_by_cases = max_confirmed_cases_bycounties[0:5]
sb.set(rc={"figure.figsize":(10,10)})
sb.barplot(x="county",y="cases",data=top_counties_by_cases,hue="county")
plt.show()


# In[19]:


#checking for relationship between cases and total population
sb.relplot(x="cases",y="Total Population",data=df)


# In[20]:


#checking for relationship between cases and total GDP
sb.relplot(x="cases",y="GDP",data=df)


# In[21]:


#plotting correlation matrix
plt.figure(figsize=(8, 8))
sb.heatmap(df.corr(),vmin=-1, vmax=1, center=0, annot=True, cmap='RdBu')


# In[22]:


#exporting the dataframe file as .csv
df.to_csv('em626.csv')


# In[23]:


#reading ths csv file with a boolean outcome variable which will be a treshold value 
df = pd.read_csv("em.csv")
print(df)


# In[24]:


#creating independent variable
Independent = ['GDP', 'Housing Units', 'Total Population']
x = df[Independent]
print(x)


# In[25]:


#creating Dependent target variable
y = df.Outcome
print(y)


# In[26]:


#importing libraries for decision tree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from graphviz import Source
import sklearn.metrics as metrics
import numpy as np


# In[27]:


#splitting the data 67% for training and 33% for testing
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.33, random_state = 1 )


# In[28]:


#creating decision tree 
clf = DecisionTreeClassifier(criterion="gini", max_leaf_nodes=10, min_samples_leaf=5, max_depth= 3)

#train decision tree
clf = clf.fit(X_train, Y_train)

#predict the model
y_Pred = clf.predict(X_test)
print(y_Pred)


# In[29]:


print("Accuracy is ", metrics.accuracy_score(Y_test, y_Pred))


# In[30]:


dot_data_exp = tree.export_graphviz(clf, out_file=None, feature_names=Independent, class_names=['0','1'], filled= True, rounded= True, special_characters= True )
graph = Source(dot_data_exp)
graph.render('df')
graph.view()


# In[31]:


#training ree
bc_tree = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train,Y_train)


# In[32]:


bc_pred = bc_tree.predict(X_test)


# In[33]:


#checking the accuracy of the decision tree
bc_tree.score(X_test, Y_test)


# In[34]:


#creating the confusion matrix in order to check TP,TN,FP,FN
cm = confusion_matrix (Y_test, bc_pred)
print(cm)


# In[35]:


#viewing confusion matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the Classifier')
fig.colorbar(cax)
ax.set_xticklabels(['a'])
ax.set_yticklabels(['b'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show


# In[36]:


#importing libraries for Random Forest
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import export_graphviz


# In[37]:


#splitting the data 67% for training and 33% for testing
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.33, random_state = 0 )


# In[38]:


rfcmodel = RandomForestClassifier(n_estimators=100, random_state=0)


# In[39]:


#training the model
rfcmodel = rfcmodel.fit(X_train, Y_train)


# In[ ]:





# In[40]:


y_Pred = rfcmodel.predict(X_test)
print(y_Pred)


# In[41]:



fn=Independent
cn=['0','1']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rfcmodel.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('rf_individualtree.png')


# In[42]:


y1_Pred = rfcmodel.predict(X_test)
print(y1_Pred)


# In[43]:


#cchecking the accuracy of the model
print("Accuracy is ", metrics.accuracy_score(Y_test, y1_Pred))


# In[44]:


fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=3000)
for index in range(0, 5):
    tree.plot_tree(rfcmodel.estimators_[index],
                   feature_names = fn, 
                   class_names=cn,
                   filled = True,
                   ax = axes[index]);
    
    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig.savefig('rf_5trees.png')


# In[45]:


#creating the confusion matrix
cm = confusion_matrix (Y_test, y1_Pred)
print(cm)


# In[46]:


#viewing the confusion matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the Classifier')
fig.colorbar(cax)
ax.set_xticklabels(['a'])
ax.set_yticklabels(['b'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show


# In[47]:


#checking the accuracy of our random forest model
rfcmodel.score(X_test, Y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




