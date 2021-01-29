
#### HR Analysis  ##########

import pandas as pd
import numpy as np


train_o = pd.read_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\train.csv')
test_o = pd.read_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\test.csv')
sub = pd.read_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\sample_submission.csv')

### dropping unnecessory columns from the data.
train = train_o.drop(columns = ['employee_id','region'])
test = test_o.drop(columns = ['employee_id','region'])



######################################################################################
#### getting missing values.
null_col = train.columns[train.isnull().any()]
null_col
train[null_col].isnull().sum()

# 1) for 'education' column
train['education'].value_counts()
train['education'] = train['education'].fillna("Bachelor's")

# 2) for 'previous_year_rating' column
train['previous_year_rating'].value_counts()
train['previous_year_rating'].mode()
train['previous_year_rating'].median()
train['previous_year_rating'] = train['previous_year_rating'].fillna(3.0)

# now for missing values in test

null_col = test.columns[test.isnull().any()]
null_col
test[null_col].isnull().sum()

# 1) for 'education' column
test['education'].value_counts()
test['education'] = test['education'].fillna("Bachelor's")


# 2) for 'previous_year_rating' column
test['previous_year_rating'].value_counts()
test['previous_year_rating'].mode()
test['previous_year_rating'].median()
test['previous_year_rating'] = test['previous_year_rating'].fillna(3.0)

########################################################################################


### checking col types,if not convert in appropriate type
train.info()
train['KPIs_met >80%'] = train['KPIs_met >80%'].astype('category')
train['awards_won?'] = train['awards_won?'].astype('category')
train['is_promoted'] = train['is_promoted'].astype('category')

test.info()
test['KPIs_met >80%'] = test['KPIs_met >80%'].astype('category')
test['awards_won?'] = test['awards_won?'].astype('category')


## categories in the columns for train and test must match.

train['department'].value_counts()
test['department'].value_counts()


train['education'].value_counts()
test['education'].value_counts()


train['gender'].value_counts()
test['gender'].value_counts()


train['recruitment_channel'].value_counts()
test['recruitment_channel'].value_counts()


train['KPIs_met >80%'].value_counts()
test['KPIs_met >80%'].value_counts()


train['awards_won?'].value_counts()
test['awards_won?'].value_counts()
# all are matching
####################################################################################


X_train = train.iloc[:,:11]
y_train = train.iloc[:,11]
X_test = test


## dummy the features
X_train = pd.get_dummies(X_train, drop_first = True)
X_test = pd.get_dummies(X_test, drop_first = True)


#######################################################################################
## data preprocessing part is over here,now applying algorithms 



