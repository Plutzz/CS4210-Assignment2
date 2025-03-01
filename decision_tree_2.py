#-------------------------------------------------------------------------
# AUTHOR: Benjamin Jiongco
# FILENAME: decision_tree2.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

     # Create dictionary maps for each feature and the class
    age = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
    spectacle = {'Myope': 1, 'Hypermetrope': 2}
    astigmatism = {'No': 1, 'Yes': 2}
    tear = {'Reduced': 1, 'Normal': 2}
    classMap = {'Yes': 1, 'No': 2}


    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here

    for row in dbTraining:
        X.append([age[row[0]], spectacle[row[1]], astigmatism[row[2]], tear[row[3]]]) 
        Y.append([classMap[row[4]]])
   
    sum = 0
    #Loop your training and test tasks 10 times here
    for i in range (10):

       #Fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
        clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       #--> add your Python code here
        dbTest = []
        #Reading the training data in a csv file
        with open("contact_lens_test.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, data in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append (data)


        TP = 0
        FP = 0
        for data in dbTest:            
            #Transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            dataX = [age[data[0]], spectacle[data[1]], astigmatism[data[2]], tear[data[3]]]
            class_predicted = clf.predict([dataX])[0]
            # print("Predicted class " + str(class_predicted))
            # print("Actual class " + str(classMap[data[4]]))
            #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            if classMap[data[4]] == class_predicted:
                TP += 1
            else:
                FP += 1
        sum += TP/(TP + FP)
    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    average = sum / 10
    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print("final accuracy when training on " + str(ds) + ": " + str(average))




