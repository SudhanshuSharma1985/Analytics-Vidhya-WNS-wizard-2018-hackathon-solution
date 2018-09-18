
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
import math
from sklearn.grid_search import GridSearchCV
import datetime as dt


df_train = pd.read_csv(r'C:\Sudhanshu\Personal\WNSHackathon\train_LZdllcl.csv',encoding='iso-8859-1')
df_test =  pd.read_csv(r'C:\Sudhanshu\Personal\WNSHackathon\test_2umaH9m.csv',encoding='iso-8859-1')

train_original=df_train.copy()
test_original=df_test.copy()
df_train.dtypes
df_test.dtypes
df_train.shape
df_test.shape
##check count of NAN
count_nan_train = len(df_train) - df_train.count()
count_nan_test = len(df_test) - df_test.count()
##Another approach to count nan
count_nan_train1  = df_train.isnull().sum()
count_nan_test1  = df_test.isnull().sum()

##Chck the column names
df_train.columns
df_test.columns

##Integer columns - employee_id,no_of_trainings,age,length_of_service,KPIs_met >80%,awards_won?,avg_training_score,is_promoted           
##float -- previous_year_rating    
## Object -- department,region,education,gender,recruitment_channel    

## univariate analysis
##Plot all categorical var
plt.figure(1)
plt.subplot(221)
df_train['department'].value_counts(normalize=True).plot.bar(figsize=(10,10), title= 'department')

plt.subplot(222)
df_train['region'].value_counts(normalize=True).plot.bar(title= 'region')

plt.subplot(223)
df_train['education'].value_counts(normalize=True).plot.bar(title= 'education')

plt.subplot(224)
df_train['recruitment_channel'].value_counts(normalize=True).plot.bar(title= 'recruitment_channel')

plt.show()  



## Bivariate Analysis

Dept=pd.crosstab(df_train['department'],df_train['is_promoted'])
Dept.div(Dept.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6,4))

Region=pd.crosstab(df_train['region'],df_train['is_promoted'])
Edu=pd.crosstab(df_train['education'],df_train['is_promoted'])
Gender=pd.crosstab(df_train['gender'],df_train['is_promoted'])
RC=pd.crosstab(df_train['recruitment_channel'],df_train['is_promoted'])

Region.div(Region.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Edu.div(Edu.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show()

Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

RC.div(RC.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()


df_train.groupby('is_promoted')['no_of_trainings'].mean().plot.bar()

df_train.columns

##Find the correlation between independent var
matrix = df_train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");

##NA imputation with mode- train dataset
df_train['education'] = df_train['education'].fillna(df_train['education'].mode()[0])
df_train['previous_year_rating'] = df_train['previous_year_rating'].fillna(df_train['previous_year_rating'].mode()[0])

##NA imputation with mode- train dataset
df_test['education'] = df_test['education'].fillna(df_test['education'].mode()[0])
df_test['previous_year_rating'] = df_test['previous_year_rating'].fillna(df_test['previous_year_rating'].mode()[0])


##Lets drop the employee_id variable as it do not have any effect on the loan status. We will do the same changes to the test dataset which we did for the training dataset.
df_train=df_train.drop('employee_id',axis=1)
df_test=df_test.drop('employee_id',axis=1)

##Segregating features and labels
X = df_train.drop('is_promoted',1)
y = df_train.is_promoted

X1 = df_test.drop('is_promoted',1)
y1 = df_test.is_promoted

##One-Hot encoding to convert Categorical var into numeric as logistics and XGB doesnt take non-numeric var
X=pd.get_dummies(X)
df_train=pd.get_dummies(df_train)
df_test=pd.get_dummies(df_test)

##train test split
from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)
pred_cv = model_lr.predict(x_cv)

accuracy_score(y_cv,pred_cv)

##93.19 accuracy on train data

##Model Evaluation using Confusion Matrix

# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_cv, pred_cv)
cnf_matrix

##Visualizing Confusion Matrix using Heatmap
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
Text(0.5,257.44,'Predicted label')



## predicting on test data now
pred_test = model_lr.predict(df_test)

submission=pd.read_csv("C:\Sudhanshu\Personal\WNSHackathon\sample_submission_M0L0uXE.csv")
submission['is_promoted']=pred_test
submission['employee_id']=test_original['employee_id']


pd.DataFrame(submission, columns=['employee_id','is_promoted']).to_csv('logistic.csv')

##applying XgBoost in the model

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = XGBClassifier(objective="binary:logistic", colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 9, alpha = 200, n_estimators = 300,num_boost_round = 2000)
model.fit(X, y)

pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)

##94.8% accuracy with XGB

pred_test = model.predict(df_test)

submission=pd.read_csv("C:\Sudhanshu\Personal\WNSHackathon\sample_submission_M0L0uXE.csv")
submission['is_promoted']=pred_test
submission['employee_id']=test_original['employee_id']
pd.DataFrame(submission, columns=['employee_id','is_promoted']).to_csv('xgb_hpt.csv')


