import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def feature_encoder(X,cols):
    # function to transform non numric data to numric data
    # be carefull if cols=('') and there's only one value 
    # it won't considered as a one row array 
    
    for c in cols:
       enc = LabelEncoder()
       enc.fit(X[c])
       X[c] = enc.transform(X[c])
    return X;

def Z_O_feature_encoder_negative(X,col):
     rows=len(X)
     for i in range(rows):
        if X.iat[i,col]=='No Negative':
            X.iat[i,col]=0
        else :
            X.iat[i,col]=1
     return X   

def Z_O_feature_encoder_positive(X,col):
     rows=len(X)
     for i in range(rows):
        if X.iat[i,col]=='No Positive':
            X.iat[i,col]=0
        else :
            X.iat[i,col]=1
     return X   

def last_column(X,col):
    rows=len(X) 
    for i in range(rows):
         print(X.iat[i,col])
         if X.iat[i,col]=='No Positive':
            X.iat[i,col]=0
         else :
            X.iat[i,col]=1
    return X   

data = pd.read_csv('Hotel_Reviews_Milestone_2.csv')
data.dropna(how='any',inplace=True)

X=data.iloc[:,:17] 


X=X.drop('lat',1)
X=X.drop('lng',1)
X=Z_O_feature_encoder_negative(X,6)
X=Z_O_feature_encoder_positive(X,9)
# make non numeric values numeric so that the pc can under stand it  
cols=('days_since_review','Review_Date','Hotel_Address','Tags','Hotel_Name','Reviewer_Nationality','Negative_Review','Positive_Review','Reviewer_Score');
X=feature_encoder(X,cols);
Y=X['Reviewer_Score']
X=X.drop('Reviewer_Score',1)
# taking our predction column and spliting the data  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True)


sc_x = StandardScaler() 
X_train = sc_x.fit_transform(X_train)  
X_test = sc_x.transform(X_test) 
  
                #KNN
                    
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_mse=metrics.mean_squared_error(knn_pred,y_test)
knn_accuracy=knn_model.score(X_test,y_test)


                          # logistic regression
LogisticRegression_model=linear_model.LogisticRegression()
LogisticRegression_model.fit(X_train,y_train) 
#prediction_log_reg_train= cls.predict(X_train)
LogisticRegression_model_Y_prediction=LogisticRegression_model.predict(X_test)
LogisticRegression_model_accuracy=LogisticRegression_model.score(X_test,y_test)
#mse_train_cls_model=metrics.mean_squared_error(y_train,prediction_cls_train)
LogisticRegression_model__mse=metrics.mean_squared_error(y_test,LogisticRegression_model_Y_prediction)
                                
                                # descision tree
                                
DecisionTree_model=tree.DecisionTreeClassifier(max_depth=8)
DecisionTree_model.fit(X_train,y_train)
DecisionTree_model_y_prediction= DecisionTree_model.predict(X_test)
DecisionTree_model_accuracy=DecisionTree_model.score(X_test,y_test)
DecisionTree_model_mse=metrics.mean_squared_error(y_test,DecisionTree_model_y_prediction)
          
                          # adaboost 
AdaBoostClassifier_model=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME.R",
                         n_estimators=100)


AdaBoostClassifier_model.fit(X_train,y_train)
AdaBoostClassifier_model_y_prediction = AdaBoostClassifier_model.predict(X_test)
AdaBoostClassifier_model_accuracy=AdaBoostClassifier_model.score(X_test,y_test)
AdaBoostClassifier_model_mse=metrics.mean_squared_error(y_test,AdaBoostClassifier_model_y_prediction)
                         
      #KNN
                    
knn_model = KNeighborsClassifier(n_neighbors=21)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_mse=metrics.mean_squared_error(knn_pred,y_test)
knn_accuracy=knn_model.score(X_test,y_test)
                               # svm
svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_train, y_train)
#svm_predictions = svm_model_linear_ovr.predict(X_test)

# model accuracy for X_test
accuracy = svm_model_linear_ovr.score(X_test, y_test)
print('One VS Rest SVM accuracy: ' + str(accuracy))
#
svm_model_linear_ovo = SVC(kernel='linear', C=1).fit(X_train, y_train)
#svm_predictions = svm_model_linear_ovo.predict(X_test)

# model accuracy for X_test
accuracy = svm_model_linear_ovo.score(X_test, y_test)
print('One VS One SVM accuracy: ' + str(accuracy))