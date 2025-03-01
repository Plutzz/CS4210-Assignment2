#-------------------------------------------------------------------------
# AUTHOR: Benjamin Jiongco
# FILENAME: knn.py
# SPECIFICATION: Complete the Python program (knn.py) to read the file email_classification.csv 
# and compute the LOO-CV error rate for a 1NN classifier on the spam/ham classification task.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 40 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
numWrongPredictions = 0
numPredictions = 0

classMap = {'ham': 1, 'spam': 2}

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#Loop your data to allow each instance to be your test set
for i in db:
    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 3], [2, 1,], ...]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    Y = []
    for row in db:
        features = []
        if row == i:
            continue
        for feature in range(len(row)-1):
            features.append(float(row[feature]))
        X.append(features)
        Y.append(float(classMap[row[len(row)-1]]))
    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here

    features = []
    for feature in range(len(i)-1):
            features.append(float(i[feature]))
    
    testSample = []
    testSample.append(features)
    print("Test sample " + str(testSample))
    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = int(clf.predict(testSample)[0])
    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != classMap[i[len(row)-1]]:
        numWrongPredictions += 1
    numPredictions += 1


#Print the error rate
#--> add your Python code here
print("=====================================================================================================================")
print("WRONG " + str(numWrongPredictions))
print("TOTAL " + str(numPredictions))
print("Error Rate " + str(numWrongPredictions/numPredictions))





