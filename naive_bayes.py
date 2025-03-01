#-------------------------------------------------------------------------
# AUTHOR: Benjamin Jiongco
# FILENAME: naive_bayes.py
# SPECIFICATION:  output the classification of each of the 10 instances from
# the file weather_test (test set) if the classification confidence is >= 0.75.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 20 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

db = []
testdb = []
X = [] 
Y = []

outlook = {'Rain': 1, 'Overcast': 2, 'Sunny': 3}
temperature = {'Cool': 1, 'Mild': 2, 'Hot': 3}
humidity = {'Normal': 1, 'High': 2}
wind = {'Strong': 1, 'Weak': 2}
classMap = {'Yes': 1, 'No': 2}

#Reading the training data in a csv file
#--> add your Python code here
#Reading the data in a csv file
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here

for row in db:
    X.append([outlook[row[1]], temperature[row[2]], humidity[row[3]], wind[row[4]]]) 
    Y.append(classMap[row[5]])
#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        testdb.append (row)

#Printing the header os the solution
#--> add your Python code here
print(testdb.pop(0))

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for sample in testdb:
    sampleNums = [outlook[sample[1]], temperature[sample[2]], humidity[sample[3]], wind[sample[4]]]
    proba = clf.predict_proba([sampleNums])[0]
    result = "Not confident enough in classification"
    if proba[0] >= 0.75:
        result = True
    elif proba[1] > 0.75:
        result = False
    print(str(sampleNums) + " Predicted Class " + str(result))

