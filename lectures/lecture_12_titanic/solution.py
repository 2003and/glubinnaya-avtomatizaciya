import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score

train_file = "/home/andrew/VS/glubinnaya-avtomatizaciya/lectures/lecture_12_titanic/train.csv"
test_file = "/home/andrew/VS/glubinnaya-avtomatizaciya/lectures/lecture_12_titanic/test.csv"
gender_submission_file = "/home/andrew/VS/glubinnaya-avtomatizaciya/lectures/lecture_12_titanic/gender_submission.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
combine = [train_df, test_df]

print(train_df.head())

X_train_prep = train_df.drop(["Survived","PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked"], axis=1)


Y_train_prep = train_df["Survived"]

print(Y_train_prep.head())

X_train_prep = pd.get_dummies(X_train_prep)
X_train_prep = X_train_prep.drop(["Sex_male",], axis=1)
X_train_prep = X_train_prep.fillna({'Age':X_train_prep.Age.median()})


# print(X_train.isnull().sum())
X_train_prep['Sex_female'] = X_train_prep["Sex_female"].astype(np.int8)
X_train_prep["Family"] = train_df["SibSp"] + train_df["Parch"]

print(X_train_prep.head())

X_train, X_test, Y_train, Y_test = train_test_split(X_train_prep, Y_train_prep, test_size=0.325, random_state=4)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, Y_train)
predict = knn.predict(X_test)
print(accuracy_score(Y_test, predict))
print(precision_score(Y_test, predict))

# = = = =

test_df_X = test_df.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked"], axis=1)
test_df_X = pd.get_dummies(test_df_X)
test_df_X = test_df_X.drop(["Sex_male",], axis=1)
test_df_X = test_df_X.fillna({'Age':test_df_X.Age.median()})
test_df_X = test_df_X.fillna({"Fare":10})
test_df_X['Sex_female'] = test_df_X["Sex_female"].astype(np.int8)
test_df_X["Family"] = test_df["SibSp"] + test_df["Parch"]
print(test_df_X.isnull().sum())

predict = knn.predict(test_df_X)
predict = pd.DataFrame(predict)
predict.to_csv("out.csv")
# predict = predict.combine(test_df["PassengerId"])
# print(type(predict))
# print(accuracy_score(Y_test, predict))
# print(precision_score(Y_test, predict))