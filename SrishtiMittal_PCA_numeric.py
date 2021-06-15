# Performing PCA on pima indians diabetes dataset

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset Description:
#    1. Number of times pregnant
#    2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#    3. Diastolic blood pressure (mm Hg)
#    4. Triceps skin fold thickness (mm)
#    5. 2-Hour serum insulin (mu U/ml)
#    6. Body mass index (weight in kg/(height in m)^2)
#    7. Diabetes pedigree function
#    8. Age (years)
#    9. Class variable (0 or 1)

path = 'SrishtiMittal.pima-indians-diabetes.csv'
dataset = np.loadtxt(path, delimiter=",")

# Here, X is assigned all the values from 1 to 8(values) and Y has the class variable(target)
# Target is to find whether a pima indian has diabetes or not using numeric dataset.
X = dataset[:,0:7]
Y = dataset[:,8]

#Assigning features values and target as well as reading the csv file
features = ['1','2','3','4','5','6','7','8','9']
df = pd.read_csv(path, names=features)              

x = df.loc[:, features].values          # Separating out the values
y = df.loc[:,['9']].values              # Separating out the target i.e the class variable

# StandardScaler standardizes a feature by subtracting the mean and then scaling to unit variance.
# The fit method is calculating the mean and variance of each of the features present in our data. 
# The transform method is transforming all the features using the respective mean and variance. 
# Here, because of the StandardScaler, the variance is unity.

sc = StandardScaler()
x_normalized = sc.fit_transform(x)

def pca(x, num_components):
      
       #Subtract the mean of each data. 
       #Axis=0 so that mean is calculate along columns.
       x_meaned= x - np.mean(x , axis=0)                      

       #Calculating the covariance matrix of the mean-centered data
       cov_mat= np.cov(x_meaned, rowvar= False)               
 
       #Calculating eigenvalues and eigenvectors of the covariance matrix using in-built function
       eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)  
 
       #Sorting the eigenvalues in descending order. 
       sorted_index= np.argsort(eigen_values)[::-1]          

       #Eigenvalues are an array with 1 dimension.
       sorted_eigenvalue = eigen_values[sorted_index] 

       #Similarly,sorting the eigenvectors.The code below implies we are selecting column-wise and that eigenvectors have 2 dimensions.      
       sorted_eigenvectors = eigen_vectors[:,sorted_index]    

       #Sorting the eigenvectors and eigenvalues helps here as the higher ones are given preference.
       #The eigenvectors with lower eigenvalues give least info about distribution of data. 
       #This helps us decide which eigenvectors to choose from according to the number of components we want.
       eigenvector_subset = sorted_eigenvectors[:,0:num_components]        
       
       #Transforming the data 
       x_reduced= np.dot(eigenvector_subset.transpose() , x_meaned.transpose()).transpose()  
 
       return x_reduced       #Result we get is the data reduced to lower dimensions from higher dimensions.

 
#Calling the pca user-defined function with 2 components.
PCA = pca(x_normalized,2)
pca_X = PCA

#Plotting the same

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

#Reshaping the scatter plot 

for color, i, target_name in zip(colors, [0, 1, 2], ['Negative', 'Positive']):
     plt.scatter(pca_X[(y == i).reshape(-1), 0],
                pca_X[(y == i).reshape(-1), 1],
                color=color, alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of pima-indians-diabetes Dataset')

# Splitting the dataset to find accuracy on training and test set

X_train, X_test, y_train, y_test = train_test_split(pca_X, y, test_size=0.8, random_state=0)
clf = RandomForestClassifier(n_estimators=100, max_depth=4)
clf.fit(X_train, y_train)
print("Accuracy on training set {}".format(clf.score(X_train, y_train)))
y_pred = clf.predict(X_test)
print("Accuracy on test set {}".format(accuracy_score(y_test, y_pred)))
