# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

from subprocess import check_output


# %%
"""
# LOAD DATASET
"""

# %%
test_df = pd.read_csv("./test.csv")
train_df = pd.read_csv("./train.csv")

# %%
train_df.head()

# %%
train_df.info()

# %%
"""
### Some values in Age/Cabin/Embarked are missing
"""

# %%
test_df.info()

# %%
"""
#### 0: dead 1 : survived

#### Pclass : 자리의 등급

#### Cabin : 선실 번호 

#### Embarked : 배를 탄 장소

#### SibSp : 함께 탑승한 형제 또는 배우자의 수

#### Parch : 함께 탑승한 부모 또는 자식의 수
"""

# %%
"""
### As train_data, age/cabin have missing parts
"""

# %%
train_df.describe()

# %%
"""
### There are 1~891 ids in train
### 892~1309 in test
"""

# %%
test_df.describe()

# %%
sns.heatmap(train_df.isnull(),cbar = False)
plt.show()
train_df.isnull()

# %%
cols = ['Survived', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']
n_rows = 2
n_cols = 3
fig, axs = plt.subplots(n_rows,n_cols, figsize = (n_rows*5,n_cols*2))
for row in range(n_rows):
    for col in range(n_cols):
        i = row*n_cols+col
        ax = axs[row][col]
        sns.countplot(train_df[cols[i]], hue=train_df["Survived"], ax=ax)
        ax.set_title(cols[i], fontsize=15, fontweight='bold')
        ax.legend(title="survived", loc='upper center') 
        
plt.tight_layout()   

# %%
"""
As you can see above, Pcclass, SibSp, Parch, Embarked,Sex are relevant with survival
"""

# %%
"""
# Data preprocessing
"""

# %%
train_df.isnull().sum() / len(train_df)*100

# %%
"""
### decision : Delete Cabin
"""

# %%
del test_df["Cabin"]
del train_df["Cabin"]

# %%
"""
#### Because test data has no survived column, 
#### we need to delete the column in train_df
"""

# %%
y_train_df = train_df.pop("Survived")
train_df.set_index('PassengerId', inplace=True)
test_df.set_index('PassengerId',inplace=True)


# %%

train_index = train_df.index
test_index = test_df.index

# %%
all_df = train_df.append(test_df)
all_df

# %%
train_df.head(1)

# %%
all_df.isnull().sum() / len(all_df)

# %%
del all_df["Name"]
del all_df["Ticket"]
all_df.head()

# %%
all_df["Sex"] = all_df["Sex"].replace({"male":0,"female":1})
all_df.head()

# %%
all_df["Embarked"].unique()

# %%
all_df["Embarked"] = all_df["Embarked"].replace({"S":0 , "C":1,"Q":0 , np.nan:99})
all_df["Embarked"].unique()

# %%


# %%
cols = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked','Fare']
for col in cols:
    print(all_df.groupby(col)["Age"].mean())

# %%
all_df.loc[(all_df["Pclass"] == 1) & (
        all_df["Age"].isnull()), "Age"] = 39.16
all_df.loc[ (all_df["Pclass"] == 2) & all_df["Age"].isnull() , "Age"] = 29.51
all_df.loc[ ( all_df["Pclass"] == 3) & all_df["Age"].isnull(), "Age"] = 24.81

# %%
all_df.isnull().sum()

# %%
cols = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked','Age']
for col in cols:
    print(all_df.groupby(col)["Fare"].mean())

# %%
all_df[all_df["Fare"].isnull()]

# %%
print(all_df.loc[(all_df["Pclass"]==1) & (all_df["Embarked"]== 0),"Fare"].mean(),
all_df.loc[(all_df["Pclass"]==1) & (all_df["Embarked"]== 1),"Fare"].mean(),
all_df.loc[(all_df["Pclass"]==1) & (all_df["Embarked"]== 99),"Fare"].mean(),
all_df.loc[(all_df["Pclass"]==2) & (all_df["Embarked"]== 0),"Fare"].mean(),
all_df.loc[(all_df["Pclass"]==2) & (all_df["Embarked"]== 1),"Fare"].mean(),
all_df.loc[(all_df["Pclass"]==2) & (all_df["Embarked"]== 99),"Fare"].mean(),
all_df.loc[(all_df["Pclass"]==3) & (all_df["Embarked"]== 0),"Fare"].mean(),
all_df.loc[(all_df["Pclass"]==3) & (all_df["Embarked"]== 1),"Fare"].mean(),
all_df.loc[(all_df["Pclass"]==3) & (all_df["Embarked"]== 99),"Fare"].mean())

# %%
all_df.loc[all_df["Fare"].isnull(), "Fare"] = 13

# %%
"""
## Modeling
"""

# %%
train_df = all_df[all_df.index.isin(train_index)]
test_df = all_df[all_df.index.isin(test_index)]


x_data = train_df.as_matrix()
y_data = y_train_df.as_matrix()

# %%
from sklearn.linear_model import LogisticRegression

# %%
model = LogisticRegression()

# %%
x_data.shape, y_data.shape

# %%
x_data

# %%
model.fit(x_data,y_data)

# %%
print(test_df.shape)
print(test_df.index)
x_test = test_df.as_matrix()
y_test = model.predict(x_test)
y_test

# %%
test_df.index

# %%
result = np.concatenate((test_df.index.values.reshape(-1,1),y_test.reshape(-1,1)),axis = 1)
result[:5]

# %%
submission = pd.DataFrame(result, columns = ["PassengerId","Survived"])
submission.head()

# %%
submission.to_csv("submission.csv", index = False)
