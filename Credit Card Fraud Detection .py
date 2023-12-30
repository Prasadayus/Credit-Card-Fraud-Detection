#!/usr/bin/env python
# coding: utf-8

# # Credit Card Kaggle Anamoly Detection

# # Context

# It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

# # Content

# The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# 
# The dataset is collected from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv('creditcard.csv')
df.head()


# # Data Inspection

# In[4]:


# Checking shape of the dataset
df.shape


# In[5]:


# Checking columns name of dataset
df.columns


# In[6]:


# Basic information of dataset
df.info()


# In[7]:


# Basic description of Dataset
df.describe()


# # Handling Duplicate Values

# In[8]:


# Checking for number of duplicated values
count_duplicated = df.duplicated().sum()
print(f'Dataset has {count_duplicated} duplicated values')


# As it has only few duplicate value so we can delete it

# In[9]:


df.drop_duplicates(inplace=True)


# In[10]:


#Now checking duplicate value again
count_duplicated = df.duplicated().sum()
print(f'Dataset has {count_duplicated} duplicated values')


# # Handling Null values

# In[11]:


# Checking for number of null values
df.isnull().sum()


# In[12]:


# Basic information of dataset
df.info()


# In[13]:


df.describe()


# # Feature Engineering

# In[14]:


## Get the Fraud and the normal dataset 

fraud = df[df['Class']==1]

normal = df[df['Class']==0]


# # Exploratory data analysis

# In[15]:


#Counting normal and fraud transaction
count_classes = pd.value_counts(df['Class'], sort=True)

# Calculating percentage frequencies
percentages = count_classes / count_classes.sum() * 100

# Plotting the distribution as a bar plot
ax = count_classes.plot(kind='bar', rot=0)

# Adding title and labels to the plot
plt.title("Transaction Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")

# Adjusting x-axis ticks and labels
plt.xticks(range(2), ['normal','fraud'])

# Annotating bars with frequency and percentage
for i, v in enumerate(count_classes):
    ax.text(i, v + 0.5, f'({percentages[i]:.2f}%)', color='black', ha='center')

# Displaying the plot
plt.show()


# In[16]:


print(fraud.shape,normal.shape)


# As we can see it has two types of transaction, featuring 473 (0.17%)frauds and 283,253(99.83%) norma transactions out of 283,726 transactions. Notably, the dataset is highly unbalanced.

# In[17]:


# We need to analyze more amount of information from the transaction data
fraud.Amount.describe()


# In[18]:


normal.Amount.describe()


# In[19]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50

ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();


# In[20]:


# We Will check Do fraudulent transactions occur more often during certain time frame ? Let us find out with a visual representation.

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# We can see it has no or very little dependency of time frame

# # Checking multicollinearity in independant variables

# In[21]:


# Plotting a correlation heatmap for the dataset
plt.figure(figsize=(30,12))
correlation=df.corr()
sns.heatmap(correlation, vmin=-1, cmap='viridis',linewidth=0.2,annot=True)
plt.show()


# # ML Model Implementation

# In this project we will use Pycaret instead of scikit-learn .PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that speeds up the experiment cycle exponentially and makes you more productive.
# 
# You can find more about pycaret on https://pycaret.org/

# In[35]:


from pycaret.classification import *
m1=setup(data=df ,target='Class' )


# In[26]:


compare_models()


# In[27]:


qda=create_model('qda')


# In[28]:


qda


# In[30]:


xgboost=create_model('xgboost')


# In[31]:


xgboost


# In[36]:


et=create_model('et')
plot_model(et, plot = 'auc')
plot_model(et, plot = 'pr')
plot_model(et, plot = 'confusion_matrix')
plot_model(et, plot = 'class_report')
plot_model(et, plot = 'feature')


# In[38]:


plot_model(xgboost, plot = 'auc')
plot_model(xgboost, plot = 'pr')
plot_model(xgboost, plot = 'confusion_matrix')
plot_model(xgboost, plot = 'class_report')
plot_model(xgboost, plot = 'feature')


# In[40]:


plot_model(qda, plot = 'auc')
plot_model(qda, plot = 'pr')
plot_model(qda, plot = 'confusion_matrix')
plot_model(qda, plot = 'class_report')


# When should you prioritize recall?
# 
# For example, when testing patients for COVID-19 it is extremely important to capture as many positive cases as possible to understand the prevalence of the virus within a given area. It is very dangerous to misdiagnose someone as not having the virus when in fact they do because they can spread the disease to others without knowing. In the opposite case, if someone is healthy but diagnosed as having the virus the penalty is they will unnecessarily self-isolate for a few days. False negatives are much more harmful than false positives in this case so we have to prioritize recall.
# 
# When should you prioritize precision?
# 
# On the other hand, when Netflix is recommending content to its users it doesn’t matter if a series the user might like isn’t displayed on the list of suggestions. What is consequential is having a high rate of suggestions that the user has no interest in. The ratio of true positives to true positives and false negatives, or recall, isn’t very important. The recommender network drives value by consistently delivering suggestions the user will enjoy so precision is the priority in this scenario because every suggestion must be a correct prediction to maintain usability.

# 
# Credit card fraud is the unauthorized use of a credit or debit card to make purchases. Credit card companies have an obligation to protect their customers’ finances and they employ fraud detection models to identify unusual financial activity and freeze a user’s credit card if transaction activity is out of the ordinary for a given individual. The penalty for mislabeling a fraud transaction as legitimate is having a user’s money stolen, which the credit card company typically reimburses. On the other hand, the penalty for mislabeling a legitimate transaction as fraud is having the user frozen out of their finances and unable to make payments. There is a very fine tradeoff between these two consequences and we will discuss how to handle this when training a model.

# In[43]:


#average cash gained if 1 more fraud is successfully flagged
avg_fraud_cost = round(np.mean(df.loc[df.Class==1].Amount.values), 1)

print(f'The average cost per uncaught fraudulent transaction is $ {avg_fraud_cost}')


# In[45]:


#average cash gained if 1 more normal transcation is successfully flagged
avg_fraud_cost = round(np.mean(df.loc[df.Class==0].Amount.values), 1)

print(f'The average cost per succesful transaction is $ {avg_fraud_cost}')


# At  $ 88.4 vs $ 123.9 per transaction, respectively, the cost ratio between type I and II errors is roughly [1:1.4].
# 
# In other words, reducing the false negative rate (catching a fraud transaction) is 1.4 times as important as reducing the false positive rate
# 
# So,we will give more importance to recall but consider precision as well.
# 
# 
# So,QuadrictDiscrimantAnaysis(qda) does very well in recall but it performs poorly in other metrics such as precision,f1 score,etc...

# # Isolation Forest Algorthim and Local Outlier Factor(LOF) Algorithm

# This are other algorithms used for outlier detection and anomaly detection. While these algorithms are available in PyCaret as unsupervised models .You can check this on https://pycaret.readthedocs.io/en/stable/api/anomaly.html .Since we have labels, we will utilize them by importing from the scikit-learn library.

# # Isolation Forest Algorithm 

# One of the newest techniques to detect anomalies is called Isolation Forests. The algorithm is based on the fact that anomalies are data points that are few and different. As a result of these properties, anomalies are susceptible to a mechanism called isolation.
# 
# This method is highly useful and is fundamentally different from all existing methods. It introduces the use of isolation as a more effective and efficient means to detect anomalies than the commonly used basic distance and density measures. Moreover, this method is an algorithm with a low linear time complexity and a small memory requirement. It builds a good performing model with a small number of trees using small sub-samples of fixed size, regardless of the size of a data set.
# 
# How Isolation Forests Work The Isolation Forest algorithm isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. The logic argument goes: isolating anomaly observations is easier because only a few conditions are needed to separate those cases from the normal observations. On the other hand, isolating normal observations require more conditions. Therefore, an anomaly score can be calculated as the number of conditions required to separate a given observation.

# In[22]:


#Creating the dataset with all dependent variables
dependent_variable = 'Class'

# Creating the dataset with all independent variables
independent_variables = list(set(df.columns.tolist()) - {dependent_variable})

# Define a random state 
state = np.random.RandomState(42)

# Create the data of independent variables
x = df[independent_variables].values
# Create the data of dependent variable
y = df[dependent_variable].values

x_outliers = state.uniform(low=0, high=1, size=(x.shape[0], x.shape[1]))
# Print the shapes of X & Y
print(x.shape)
print(y.shape)


# In[23]:


from sklearn.model_selection import train_test_split
#Creating train validation and test data
x_train,x_rem,y_train,y_rem=train_test_split(x,y, train_size=0.75,random_state=1)
x_valid,x_test,y_valid,y_test=train_test_split(x,y, test_size=0.40,random_state=1)


# In[24]:


outlier_fraction = len(fraud)/float(len(normal))


# In[25]:


from pycaret.anomaly import *
exp_name = setup(data = x_train)
iforest = create_model('iforest',fraction=outlier_fraction)
iforest_predictions = predict_model(model = iforest, data = x_valid)


# In[26]:


plot_model(iforest)


# In[ ]:


lof = create_model('lof',fraction=outlier_fraction)
lof_predictions = predict_model(model = lof, data = x_valid)


# In[28]:


plot_model(lof)


# In[29]:


from sklearn.metrics import classification_report,accuracy_score
n_errors = (iforest_predictions['Anomaly'] != y_valid).sum()
# Run Classification Metrics
print("The total error in validation set of Iforest model is:",n_errors)
print("Accuracy Score :")
print(accuracy_score(y_valid,iforest_predictions['Anomaly']))
print("Classification Report :")
print(classification_report(y_valid,iforest_predictions['Anomaly']))


# In[33]:


n_errors = (lof_predictions['Anomaly'] != y_valid).sum()
# Run Classification Metrics
print("The total error in validation set of Lof model is:",n_errors)
print("Accuracy Score :")
print(accuracy_score(y_valid,lof_predictions['Anomaly']))
print("Classification Report :")
print(classification_report(y_valid,lof_predictions['Anomaly']))


# # Final evaluation on test dataset

# In[32]:


iforest_final_predictions = predict_model(model = iforest, data = x_test)


# In[35]:


from sklearn.metrics import classification_report,accuracy_score
n_errors = (iforest_final_predictions['Anomaly'] != y_test).sum()
# Run Classification Metrics
print("The total error in test set of Iforest model is:",n_errors)
print("Accuracy Score :")
print(accuracy_score(y_test,iforest_final_predictions['Anomaly']))
print("Classification Report :")
print(classification_report(y_test,iforest_final_predictions['Anomaly']))


# # Observations :

# Isolation Forest has a 99.76% more accurate than LOF of 99.73%.
# 
# But mainly when comparing error precision & recall for 2 models , the Isolation Forest performed much better than the LOF 

# # Final Observation:

# Overall the XGBoost perfomed very well followed ExtraTreesClassifier which is mch better than outlier detection method Lof and Iforest

# In[ ]:




