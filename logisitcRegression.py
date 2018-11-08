# ===============================================================================
# Project Details
# ===============================================================================
# Author    : Kevin Sequeira
# Date      : 11-October-2018
# Project   : Logistic Regression
# ===============================================================================
# Set the Working Directory for the Project
# ===============================================================================
print('\n' * 100);
import os;
os.chdir('D:/Programming/Machine Learning/Machine Learning with Python/Logistic Regression/');
print('Current Working Directory:');
print(os.getcwd());
print();
# ===============================================================================
# Import all the necessary Packages
# ===============================================================================
from subprocess import call;
call("pip install pandas", shell = True);
import pandas as pan;
call("pip install numpy", shell = True);
import numpy as np;
call("pip install matplotlib", shell = True);
import matplotlib.pyplot as plt;
call("pip install sklearn", shell = True);
from sklearn.linear_model import LogisticRegression;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import mean_squared_error, r2_score;
from sklearn import preprocessing;
from sklearn.feature_selection import RFE;
from sklearn.metrics import confusion_matrix;
from sklearn.metrics import classification_report;
call("pip install imblearn", shell = True);
from imblearn.over_sampling import SMOTE;
call("pip install statsmodels", shell = True);
import statsmodels.api as statsModels;
# ===============================================================================
# Import the Dataset for the Project
# ===============================================================================
bankingData = pan.read_csv('banking.csv', header = 0);
bankingData = bankingData.dropna();
print('Dimensions of Banking Dataset: ', bankingData.shape);
print('List of Column Headers: ');
bankingDataColumns = list(bankingData.columns);
print(bankingDataColumns);
print('Banking Data Table: ');
print(bankingData.head);
print();
# ===============================================================================
# Provide Statistical Summary for the Dataset
# ===============================================================================
bankingStatisticalSummary = pan.DataFrame(columns = ['Column_Name', 'Data_Type', 'Count', 'Values']);
bankingStatisticalSummary['Column_Name'] = bankingDataColumns;
bankingStatisticalSummary['Data_Type'] = 'Categorical';
for column in bankingDataColumns:
    if np.issubdtype(bankingData[column].dtype, np.number):
        bankingStatisticalSummary['Data_Type'][bankingStatisticalSummary.Column_Name == column] = 'Numerical';
    bankingStatisticalSummary['Count'][bankingStatisticalSummary.Column_Name == column] = len(bankingData[column]);
    bankingStatisticalSummary['Values'][bankingStatisticalSummary.Column_Name == column] = len(bankingData[column].unique());
print('Banking Data Statistical Summary: ');
print(bankingStatisticalSummary);
print();
# ===============================================================================
# Clean and group the data in the 'Education' column
# ===============================================================================
print('Data Values in the Education Column: ');
print(bankingData['education'].unique());
print();

bankingData['education'] = np.where(bankingData['education'] == 'basic.4y', 'basic', bankingData['education']);
bankingData['education'] = np.where(bankingData['education'] == 'basic.6y', 'basic', bankingData['education']);
bankingData['education'] = np.where(bankingData['education'] == 'basic.9y', 'basic', bankingData['education']);
print('New Data Values for the Education Column: ');
print(bankingData['education'].unique());
print();
# ===============================================================================
# Create Dummy Variables for the following Categorical Variables:
#   1. job
#   2. marital
#   3. education
#   4. default
#   5. housing
#   6. loan
#   7. contact
#   8. month
#   9. day_of_week
#   10. poutcome
# ===============================================================================
categoricalVariables = ['job',
                        'marital',
                        'education',
                        'default',
                        'housing',
                        'loan',
                        'contact',
                        'month',
                        'day_of_week',
                        'poutcome'];
bankingData = pan.get_dummies(bankingData);
for variable in categoricalVariables:
    columns = [column for column in bankingData.columns if variable in column]
    print('Dummy Columns for', variable, ':');
    print(columns);
    referenceColumn = bankingData[columns].sum().sort_values(ascending = False).index.values[0]
    print('Reference Column: ', referenceColumn);
    print();
    bankingData = bankingData.drop(columns = [referenceColumn], axis = 1);
print('New Columns for Banking Dataset: ');
print(list(bankingData.columns));
print();
# ===============================================================================
# Up-sample the minority Target Class using SMOTE (Synthetic Minority
# Oversampling Technique)
# ===============================================================================
bankingTarget = bankingData.loc[:, bankingData.columns == 'subscription'];
bankingData = bankingData.loc[:, bankingData.columns != 'subscription'];

oversampling = SMOTE(random_state = 0);
bankingDataTrain, bankingDataTest, bankingTargetTrain, bankingTargetTest\
    = train_test_split(bankingData, bankingTarget, test_size = 0.3, random_state = 0)
columns = bankingDataTrain.columns.values;

print('Banking Training Data Size: ');
print(bankingDataTrain.shape);
print();

print('Banking Test Data Size: ');
print(bankingDataTest.shape);
print();

oSBankingDataTrain, oSBankingTargetTrain = oversampling.fit_sample(bankingDataTrain, bankingTargetTrain);
oSBankingDataTrain = pan.DataFrame(data = oSBankingDataTrain, columns = columns);
oSBankingTargetTrain = pan.DataFrame(data = oSBankingTargetTrain, columns = ['subscription']);

print('Length of Oversampled Data is: ', len(oSBankingTargetTrain));
print('Number of "Subscription = 0" cases in Oversampled Data: ',
      len(oSBankingTargetTrain[oSBankingTargetTrain['subscription'] == 0]));
print('Number of "Subscription = 1" cases in Oversampled Data: ',
      len(oSBankingTargetTrain[oSBankingTargetTrain['subscription'] == 1]));
print('Proportion of "Subscription = 0" cases in Oversampled Data: ',
      len(oSBankingTargetTrain[oSBankingTargetTrain['subscription'] == 0])/len(oSBankingTargetTrain));
print('Proportion of "Subscription = 1" cases in Oversampled Data: ',
      len(oSBankingTargetTrain[oSBankingTargetTrain['subscription'] == 1])/len(oSBankingTargetTrain));
print();

print('Oversampled Banking Training Data Size: ');
print(oSBankingDataTrain.shape);
print();

print('Banking Test Data Size: ');
print(bankingDataTest.shape);
print();
# ===============================================================================
# Use RFE (Recursive Feature Elimination) to choose the best features for
# estimating the model.
# ===============================================================================
columns = oSBankingDataTrain.columns.values;
target = ['subscription'];

logReg = LogisticRegression();
rfe = RFE(logReg, 20)
rfe = rfe.fit(X = oSBankingDataTrain, y = oSBankingTargetTrain.values.ravel());
print('Training Data Columns: ');
print(oSBankingDataTrain.columns.values);
print('Column Ranking for Estimating Model: ');
columnRanks = rfe.ranking_;
print(columnRanks)

selectedColumns = [column for column, rank in zip(columns, columnRanks) if rank == 1];
print('Selected Columns: ', len(selectedColumns));
print(selectedColumns);
print();

oSBankingDataTrain = oSBankingDataTrain[selectedColumns]
# ===============================================================================
# Implement the Logistic Regression Model using the selected columns
# ===============================================================================
logitModel = statsModels.Logit(oSBankingTargetTrain, oSBankingDataTrain);
logitResults = logitModel.fit();
print(logitResults.summary2());
print();

# The following lines are to be used if we are not removing most frequent dummy
# variables
# oSBankingDataTrain = oSBankingDataTrain.drop(columns = ['default_no',
#                                                        'default_unknown',
#                                                        'contact_cellular',
#                                                        'contact_telephone'],
#                                             axis = 1);
finalPredictorColumns = oSBankingDataTrain.columns.values;
print('Final Predictor Columns: ', finalPredictorColumns);
print();

logitModel = statsModels.Logit(oSBankingTargetTrain, oSBankingDataTrain);
logitResults = logitModel.fit();
print(logitResults.summary2());
print();
# ===============================================================================
# Fit the Logistic Regression Model using the final predictor columns
# ===============================================================================
logReg = LogisticRegression();
logReg.fit(oSBankingDataTrain, oSBankingTargetTrain);
print('Model Intercept: ', logReg.intercept_);
print('Model Coefficients: ');
print(logReg.coef_.T);
print();
# ===============================================================================
# Predict the Target Class using the Logistic Regression Model
# ===============================================================================
bankingDataTest = bankingDataTest[finalPredictorColumns];
bankingTargetPredicted = logReg.predict(bankingDataTest);
print('Accuracy of Logistic Regression Classifier on Test Set: ', logReg.score(bankingDataTest, bankingTargetTest));
print();
# ===============================================================================
# Print the Confusion Matrix for the Model Prediction
# ===============================================================================
confusionMatrix = confusion_matrix(bankingTargetTest, bankingTargetPredicted);
print('Confusion Matrix for the Model: ');
print(confusionMatrix);
print();
# ===============================================================================
# Print the Precision, Recall, F-Measure, and Support for the Model
# ===============================================================================
classificationReport = classification_report(bankingTargetTest, bankingTargetPredicted);
print('Classification Report: ');
print(classificationReport);
# ===============================================================================
