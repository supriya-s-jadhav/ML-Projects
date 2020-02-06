# Load libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
data = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris = pd.read_csv(data,names = names)

# Explore dataset
print("Explore dataset\n")
#print("\n\nNumber of rows and columns: ", iris.shape)
print(iris.info())
print("\nClass distributuion: \n", iris['class'].value_counts())
print("\nPeek at the dataset:\n",iris.head(5))
print("\nStatistical summary of the dataset:\n",iris.describe())

# Data Visualization

# 1.Plot the relation of each feature with each species
plt.xlabel('Features')
plt.ylabel('Species')

pltX = iris.loc[:, 'sepal-length']
pltY = iris.loc[:,'class']
plt.scatter(pltX, pltY, color='blue', label='sepal-length')

pltX = iris.loc[:, 'sepal-width']
pltY = iris.loc[:,'class']
plt.scatter(pltX, pltY, color='green', label='sepal-width')

pltX = iris.loc[:, 'petal-length']
pltY = iris.loc[:,'class']
plt.scatter(pltX, pltY, color='red', label='petal-length')

pltX = iris.loc[:, 'petal-width']
pltY = iris.loc[:,'class']
plt.scatter(pltX, pltY, color='black', label='petal-width')

plt.legend(loc=4, prop={'size':8})
plt.show()

# 2.Box and whisker plots
iris.plot(kind='box', subplots=True, layout=(2,2),sharex=False, sharey=False)
plt.show()

# 3.Histograms
iris.hist()
plt.show()

# 4.Multivariate plots
# Scatter plot
pd.plotting.scatter_matrix(iris)
plt.show()

# 5.Correlation matrix and correlation plot
corr = iris.corr()
corr.style.background_gradient(cmap='coolwarm')

# Model Building

# 1.Split-out validation data set
temp_dataset = iris.values
X = temp_dataset[:, 0:4]
Y = temp_dataset[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = 0.2) #splitting in 80-20 train-validate

# 2. Use stratified 10-fold CV
# NOTE: stratification is generally a better scheme, both in terms of bias and variance, when compared to regular cross-validation.
# 3. Build Models
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
random.seed(0)
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
print("\nFinal model selected is LDA with highest model accuracy of 98.33%.\n")

# Select best Model with highest model accuracy and make the predictions
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
Y_prediction = model.predict(X_validation)
compare_values = pd.DataFrame({'Actual': Y_validation.flatten(), 'Predicted': Y_prediction.flatten()})
print(compare_values)
print()
# Evaluate predictions
print("Model accuracy is: ")
print(accuracy_score(Y_validation, Y_prediction))
print(confusion_matrix(Y_validation, Y_prediction))
print(classification_report(Y_validation, Y_prediction))

