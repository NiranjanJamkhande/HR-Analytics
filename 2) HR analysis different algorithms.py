




### Logistic  ######
# Import the necessary modules
from sklearn.linear_model import LogisticRegression


# Create the classifier: logreg
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg.fit(X_train,y_train)
# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)
y_pred = pd.DataFrame(y_pred)

submission = pd.concat([sub['employee_id'],y_pred],keys = ["employee_id","is_promoted"],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\submission.csv')


##### now SGD Classifier  ########################################################

from sklearn.linear_model import SGDClassifier
sgdClass = SGDClassifier(loss='log')
sgdClass.fit(X_train,y_train)
y_pred = sgdClass.predict(X_test)
y_pred = pd.DataFrame(y_pred)


submission = pd.concat([sub['employee_id'],y_pred],keys = ["employee_id","is_promoted"],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\sub-sgd.csv')

#################### LDA ######################################


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
da = LinearDiscriminantAnalysis()

da.fit(X_train,y_train)
y_pred = da.predict(X_test)
y_pred = pd.DataFrame(y_pred)



submission = pd.concat([sub['employee_id'],y_pred],keys = ["employee_id","is_promoted"],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\sub-LDA.csv')

############### QDA  ##################################

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
da = QuadraticDiscriminantAnalysis()

da.fit(X_train,y_train)
y_pred = da.predict(X_test)
y_pred = pd.DataFrame(y_pred)


submission = pd.concat([sub['employee_id'],y_pred],keys = ["employee_id","is_promoted"],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\sub-QDA.csv')



################### SVM  #####################################

from sklearn.svm import SVC

svc = SVC(probability = True,kernel='linear',verbose=2)
fitSVC = svc.fit(X_train, y_train)
y_pred = fitSVC.predict(X_test)


########decision tree #####
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=2020)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
y_pred = pd.DataFrame(y_pred)



submission = pd.concat([sub['employee_id'],y_pred],keys = ["employee_id","is_promoted"],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\sub-DTree.csv')


#######  Random Forest  #####

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(random_state=1211,
                                  n_estimators=500,oob_score=True)
model_rf.fit( X_train , y_train )
y_pred = model_rf.predict(X_test)
y_pred = pd.DataFrame(y_pred)



submission = pd.concat([sub['employee_id'],y_pred],keys = ["employee_id","is_promoted"],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\sub-rf.csv')



##### Bagging  #######

from sklearn.ensemble import BaggingClassifier

model_rf = BaggingClassifier(random_state=1211,oob_score=True,
                             max_features=X_train.shape[1],
                             n_estimators=50,max_samples=X_train.shape[0])

model_rf.fit( X_train , y_train )
y_pred = model_rf.predict(X_test)
y_pred = pd.DataFrame(y_pred)



submission = pd.concat([sub['employee_id'],y_pred],keys = ["employee_id","is_promoted"],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\sub-bagging.csv')

####### 


model_rf = BaggingClassifier(base_estimator = RandomForestClassifier(random_state=1211,
                                  n_estimators=500,oob_score=True) ,
                             random_state=1211,oob_score=True,
                             max_features=X_train.shape[1],
                             n_estimators=50,max_samples=X_train.shape[0])
                             
model_rf.fit( X_train , y_train )
y_pred = model_rf.predict(X_test)

##### Voting ###


from sklearn.linear_model import SGDClassifier
sgdClass = SGDClassifier(loss='log')


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
da = QuadraticDiscriminantAnalysis()


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state=1211, n_estimators=500,oob_score=True)





from sklearn.ensemble import VotingClassifier
Voting = VotingClassifier(estimators=[('SGD',sgdClass),
                                      ('DA',da),
                                      ('RF',model_rf)],voting='soft')


Voting.fit(X_train,y_train)
y_pred = Voting.predict(X_test)
y_pred = pd.DataFrame(y_pred)



submission = pd.concat([sub['employee_id'],y_pred],keys = ["employee_id","is_promoted"],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\sub-voting.csv')

###

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
da = QuadraticDiscriminantAnalysis()


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state=1211, n_estimators=500,oob_score=True)


from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(random_state=1211,oob_score=True,
                             max_features=X_train.shape[1],
                             n_estimators=50,max_samples=X_train.shape[0])



from sklearn.ensemble import VotingClassifier
Voting = VotingClassifier(estimators=[('DA',da),
                                      ('RF',model_rf),
                                      ('BAG',bag)],voting='soft')


Voting.fit(X_train,y_train)
y_pred = Voting.predict(X_test)
y_pred = pd.DataFrame(y_pred)



submission = pd.concat([sub['employee_id'],y_pred],keys = ["employee_id","is_promoted"],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\sub-voting_2.csv')

#### Gradient Boosting 

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(random_state=1200)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
y_pred = pd.DataFrame(y_pred)



submission = pd.concat([sub['employee_id'],y_pred],keys = ["employee_id","is_promoted"],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\sub-GB.csv')

########### XGBoost   ###########

from xgboost import XGBClassifier

clf = XGBClassifier(random_state=2000)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
y_pred = pd.DataFrame(y_pred)



submission = pd.concat([sub['employee_id'],y_pred],keys = ["employee_id","is_promoted"],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\sub-XGB.csv')

###########  Stacking    ########


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
da = QuadraticDiscriminantAnalysis()


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state=1211,
                                  n_estimators=500,oob_score=True)


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(random_state=1200)



from sklearn.linear_model import SGDClassifier
sgdClass = SGDClassifier(loss='log')




models_considered = [
                     ('DA', da),
                     ('RF',model_rf),
                     ('GBC',clf),
                     ('SGD',sgdClass)]



from sklearn.ensemble import VotingClassifier
Voting = VotingClassifier(estimators=[('DA',da),
                                      ('RF',model_rf),
                                      ('BAG',bag)],voting='soft')



from sklearn.ensemble import StackingClassifier
stack = StackingClassifier(estimators = models_considered,
                           final_estimator = Voting,
                           stack_method="predict",
                           passthrough=True,verbose = 2)

stack.fit(X_train,y_train)
y_pred = stack.predict(X_test)
y_pred = pd.DataFrame(y_pred)



submission = pd.concat([sub['employee_id'],y_pred],keys = ["employee_id","is_promoted"],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\sub-stack_8.csv')

######################################################
import catboost as ctb


cbc = ctb.CatBoostClassifier(random_state=1200, depth = 3)
cbc.fit(X_train,y_train)

y_pred = cbc.predict(X_test)
y_pred = pd.DataFrame(y_pred)



submission = pd.concat([sub['employee_id'],y_pred],keys = ["employee_id","is_promoted"],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\sub-catboost_2.csv')

########################### ANN   ########################################

from sklearn.neural_network import MLPClassifier


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(9,7,4,2),activation='logistic',
                    random_state=2018, verbose = 2,max_iter=100)


mlp.fit( X_train , y_train )
y_pred = mlp.predict(X_test)
y_pred = pd.DataFrame(y_pred)



submission = pd.concat([sub['employee_id'],y_pred],keys = ["employee_id","is_promoted"],
                                         axis = "columns")


submission.to_csv('C:\\Users\\Admin\\Desktop\\HR analytics\\sub-ANN_2.csv')
##################################################################################
