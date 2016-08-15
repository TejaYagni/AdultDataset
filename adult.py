

#Import required libraries
import numpy as np
import pandas as pd

#Reads the training and testing datasets
train_adult = pd.read_csv('/home/tezaa/Documents/GitHub/Adult Dataset/adult.csv')
test_adult = pd.read_csv('/home/tezaa/Documents/GitHub/Adult Dataset/adult_test.csv')


#Creating Dataframes for train and test data
train_df = pd.DataFrame(train_adult,columns = ['Age','WorkClass','fnlwgt','education','education-num','marital-status','Occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary'])
test_df = pd.DataFrame(test_adult,columns = ['Age','WorkClass','fnlwgt','education','education-num','marital-status','Occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary'])


#Creating Data frames for train and test data by removing salary column
train_nolabel_df = pd.DataFrame(train_df, columns = ['Age','WorkClass','fnlwgt','education','education-num','marital-status','Occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country'])
test_nolabel_df = pd.DataFrame(test_df, columns = ['Age','WorkClass','fnlwgt','education','education-num','marital-status','Occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country'])


# Discretizing hours per week into 5
train_nolabel_df['hours-per-week'] = pd.cut(train_nolabel_df['hours-per-week'], 5 , labels = [1,2,3,4,5])
test_nolabel_df['hours-per-week'] = pd.cut(test_nolabel_df['hours-per-week'], 5 , labels = [1,2,3,4,5])


#Discretizing capital-gain into 8
train_nolabel_df['capital-gain'] = pd.cut(train_nolabel_df['capital-gain'], 8 , labels = [1,2,3,4,5,6,7,8])
test_nolabel_df['capital-gain'] = pd.cut(test_nolabel_df['capital-gain'], 8 , labels = [1,2,3,4,5,6,7,8])


#Discretizing capital-loss into 8
train_nolabel_df['capital-loss'] = pd.cut(train_nolabel_df['capital-loss'], 8 , labels = [1,2,3,4,5,6,7,8])
test_nolabel_df['capital-loss'] = pd.cut(test_nolabel_df['capital-loss'], 8 , labels = [1,2,3,4,5,6,7,8])


#importing packages from sklearn classification and processing of data to convert the values in string to integer so that data suits the classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


#Seperate DataFrame created for salary which is passed into predict method for classification
label_df = pd.DataFrame(train_df['salary'])

#Creating arrays with unique values that each column has
country_array = train_nolabel_df['native-country'].unique() 
sex_array = train_nolabel_df['sex'].unique()
race_array = train_nolabel_df['race'].unique()
relationship_array = train_nolabel_df['relationship'].unique()
occupation_array = train_nolabel_df['Occupation'].unique()
marital_array = train_nolabel_df['marital-status'].unique()
edu_array = train_nolabel_df['education'].unique()
work_array = train_nolabel_df['WorkClass'].unique()


#Transforming all the string values in columns to integer values
country_le = preprocessing.LabelEncoder()
country_le.fit(country_array)
country_le.classes_
train_nolabel_df['native-country'] = country_le.transform(train_nolabel_df['native-country'])

sex_le = preprocessing.LabelEncoder()
sex_le.fit(sex_array)
train_nolabel_df['sex'] = sex_le.transform(train_nolabel_df['sex'])

race_le = preprocessing.LabelEncoder()
race_le.fit(race_array)
train_nolabel_df['race'] = race_le.transform(train_nolabel_df['race'])
race_le.classes_

relation_le = preprocessing.LabelEncoder()
relation_le.fit(relationship_array)
train_nolabel_df['relationship'] = relation_le.transform(train_nolabel_df['relationship'])

occupation_le = preprocessing.LabelEncoder()
occupation_le.fit(occupation_array)
train_nolabel_df['Occupation'] = occupation_le.transform(train_nolabel_df['Occupation'])

marital_le = preprocessing.LabelEncoder()
marital_le.fit(marital_array)
train_nolabel_df['marital-status'] = marital_le.transform(train_nolabel_df['marital-status'])

edu_le = preprocessing.LabelEncoder()
edu_le.fit(edu_array)
train_nolabel_df['education'] = edu_le.transform(train_nolabel_df['education'])

work_le = preprocessing.LabelEncoder()
work_le.fit(work_array)
train_nolabel_df['WorkClass'] = work_le.transform(train_nolabel_df['WorkClass'])


#Deleting the column 'fnlwgt' from dataframe as it is of no use for classification
del train_nolabel_df['fnlwgt']


#Dicretizing values in Age column to 7
train_nolabel_df['Age'] = pd.cut(train_nolabel_df['Age'],7,labels = [1,2,3,4,5,6,7] )


#Converting the data in test file from string to int just as I did in train data
test_nolabel_df['native-country'] = country_le.transform(test_nolabel_df['native-country'])

test_nolabel_df['sex'] = sex_le.transform(test_nolabel_df['sex'])

test_nolabel_df['race'] = race_le.transform(test_nolabel_df['race'])

test_nolabel_df['relationship'] = relation_le.transform(test_nolabel_df['relationship'])

test_nolabel_df['Occupation'] = occupation_le.transform(test_nolabel_df['Occupation'])

test_nolabel_df['marital-status'] = marital_le.transform(test_nolabel_df['marital-status'])

test_nolabel_df['education'] = edu_le.transform(test_nolabel_df['education'])

test_nolabel_df['WorkClass'] = work_le.transform(test_nolabel_df['WorkClass'])


#Deleting column fnlwgt as I dont need it
del test_nolabel_df['fnlwgt']


#Building a classifier
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(train_nolabel_df,label_df)


#predicting the test data using classifier
pred = rf.predict(test_nolabel_df)


#saving the predicted values into a text file
np.savetxt('adult1.csv',np.c_[range(1,16282),pred],delimiter=',',header = 'adultId,Salary',comments = '',fmt = '%s')


#Actual values in test file
actual_values = np.array(test_df['salary'])
actual_values


#Since the actual values have a dot at the end, I removed them to suit the predicted values
modified_actual_values = []
for i in actual_values:
    modified_actual_values.append(i[:len(i)-1])


#Creating an array and storing values
numpy_actual_values = np.array(modified_actual_values)

#Method to give precision 
def prec(pred,numpy_actual_values):
    output = pred == numpy_actual_values
    count = np.sum(output)
    precision = float(count)/float(output.size)
    print (precision)

#Method call
print "\nRandom Forest Classifier: "
prec(pred,numpy_actual_values)

#import for AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

#Training the classifier
abc = AdaBoostClassifier(n_estimators = 100)
abc.fit(train_nolabel_df,label_df)

#Predicting the class for test data
pred_ada = abc.predict(test_nolabel_df)

#Storing the output set to file
np.savetxt('adult_ada.csv',np.c_[range(1,16282),pred],delimiter=',',header = 'adultId,Salary',comments = '',fmt = '%s')

#Printing output to console
print "\nAda Boost Classifier: "
prec(pred_ada,numpy_actual_values)

#Importing Classifier
from sklearn.tree import DecisionTreeClassifier

#Training the classifier
dtc = DecisionTreeClassifier(random_state = 0)
dtc.fit(train_nolabel_df,label_df)
pred_dtc = dtc.predict(test_nolabel_df)

#Storing the output set to file
np.savetxt('adult_dtc.csv',np.c_[range(1,16282),pred],delimiter=',',header = 'adultId,Salary',comments = '',fmt = '%s')

#Printing output to console
print "\nDecision Tree Classifier: "
prec(pred_dtc,numpy_actual_values)

