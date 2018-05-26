# Import the neccessary modules for data manipulation and visual representation
import pandas as pd
import numpy as np
import heapq
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn import tree,preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
#Read the analytics csv file and store our dataset into a dataframe called "df"
df=pd.read_csv('kddcup.data_10_percent_corrected',header=None)

#claim features' name
df.columns=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment'\
,'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root'\
,'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login'\
,'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate'\
,'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate'\
,'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate'\
,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label']

#feature extraction
#transform symbolic into numeric
df["protocol_type"] = df["protocol_type"].astype('category').cat.codes
df["service"] = df["service"].astype('category').cat.codes
df["flag"] = df["flag"].astype('category').cat.codes
df["land"] = df["land"].astype('category').cat.codes
df["logged_in"] = df["logged_in"].astype('category').cat.codes
df["is_host_login"] = df["is_host_login"].astype('category').cat.codes
df["is_guest_login"] = df["is_guest_login"].astype('category').cat.codes

#label target
#We define all attacks type as label '1', normal statue as label '0' 
df.loc[df['label']!='normal.','label']=1 #attack
df.loc[df['label']=='normal.','label']=0 #normal


#divide dataset into training set and testing set
target_label = df['label']
df.drop(labels=['label'],axis=1,inplace=True)
df.insert(0,'label',target_label)
df.insert(1,'augmentation',1)

X=df.drop('label',axis=1)
y=df['label']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,stratify=y) #stratify keep the ratio of classes
y_train=y_train.astype(int)
y_test=y_test.astype(int)

#preprocessing
scaler=preprocessing.StandardScaler().fit(X_train) #Standardization
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

print("Prediction based on FULL FAETURES")

#LogisticRegression model
model=LogisticRegression(penalty='l2',tol=1e-4,C=1,max_iter=100)
model.fit(X_train_scaled,y_train)
ACU_train=model.score(X_train_scaled,y_train)
ACU_test=model.score(X_test_scaled,y_test)
print("LogisticRegression Classifier:")
print("Train acurracy:%.2f%%"%(ACU_train*100))
print("Test acurracy:%.2f%%"%(ACU_test*100))

#Decision Tree model
#training model
CART=tree.DecisionTreeClassifier()
model=CART.fit(X_train_scaled,y_train)
#prediction
ACU_train=model.score(X_train_scaled,y_train)
ACU_test=model.score(X_test_scaled,y_test)
print("DecisionTree Classifier:")
print("Train acurracy:%.2f%%"%(ACU_train*100))
print("Test acurracy:%.2f%%"%(ACU_test*100))
tmp=model.feature_importances_
#print(tmp)
index=heapq.nlargest(10,range(len(tmp)),tmp.take)
index=sorted(index)

#Random Forest model
#training model
model=RandomForestClassifier()
model.fit(X_train_scaled,y_train)
#prediction
ACU_train=model.score(X_train_scaled,y_train)
ACU_test=model.score(X_test_scaled,y_test)
print("Random Forest Classifier:")
print("Train acurracy:%.2f%%"%(ACU_train*100))
print("Test acurracy:%.2f%%"%(ACU_test*100))


#feature selection
#X_new=SelectKBest(chi2,k=10).fit_transform(X,y.astype(int))
X_new=X.ix[:,index]
#tranform into polynomial features
#poly=preprocessing.PolynomialFeatures(2)
#X_new=poly.fit_transform(X_new)
#pca=PCA(n_components=10)
#pca.fit(X)
#X_new=pca.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_new,y,test_size=0.15,stratify=y) #stratify keep the ratio of classes
y_train=y_train.astype(int)
y_test=y_test.astype(int)


print("Prediction based on 10 FAETURES")
print("Selected Features:"+str(sorted(index)))

#LogisticRegression model
model=LogisticRegression(penalty='l2',tol=1e-4,C=1,max_iter=100)
model.fit(X_train,y_train)
ACU_train=model.score(X_train,y_train)
ACU_test=model.score(X_test,y_test)
print("LogisticRegression Classifier:")
print("Train acurracy:%.2f%%"%(ACU_train*100))
print("Test acurracy:%.2f%%"%(ACU_test*100))

#Decision Tree model
#training model
CART=tree.DecisionTreeClassifier(max_depth=40) #optimal max_depth/max_leaf_nodes  should be determined by cross-validation
model=CART.fit(X_train,y_train)
#prediction
ACU_train=model.score(X_train,y_train)
ACU_test=model.score(X_test,y_test)
print("DecisionTree Classifier:")
print("Train acurracy:%.2f%%"%(ACU_train*100))
print("Test acurracy:%.2f%%"%(ACU_test*100))
#print(model.feature_importances_)

#Random Forest model
#training model
model=RandomForestClassifier(max_depth=40) #optimal max_depth/max_leaf_nodes  should be determined by cross-validation
model.fit(X_train,y_train)
#prediction
ACU_train=model.score(X_train,y_train)
ACU_test=model.score(X_test,y_test)
print("Random Forest Classifier:")
print("Train acurracy:%.2f%%"%(ACU_train*100))
print("Test acurracy:%.2f%%"%(ACU_test*100))
