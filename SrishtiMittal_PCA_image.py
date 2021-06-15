# Leeds Butterfly Dataset
# Performing PCA on Butterfly image dataset
# Originally consists of 832 images of mostly 256*256 dimensions. 
# Attaching 5 images of the same for trial of code

import numpy as np
from random import randrange
import os
from PIL import Image
import matplotlib.pyplot as plt
from numpy import linalg
import sys

# Here, picking up a random image from the dataset file in C drive,converting image to greyscale and showing original image
# L mode is normally interpreted as grayscale.
filenames = os.listdir('C:/dataset')
img = Image.open('C:/dataset/' + filenames[randrange(len(filenames))]).convert('L').resize((256,256))
plt.imshow(img,cmap='gray')
plt.show()

def pca(img,vals):
  
  #Convert image to numpy array
  #getdata is to returns the contents of this image as a sequence object containing pixel values. 
  #band 0 here supports white , while band 1 supports black. By default, band returns all bands in image
  imgmat = np.array(list(img.getdata(band=0)),float)

  #reshape according to original image dimensions/ convert to matrix
  #0 for along x-axis(width) and 1 for along y-axis(height)
  imgmat.shape = (img.size[0],img.size[1])

  #Calculating covariance matrix. Each column represents a feature and each row represents a sample.
  #Axis=1 so that mean is calculate along columns.
  cov_mat = imgmat - np.mean(imgmat,axis=1)


  #Calculating eigenvalues and eigenvectors of the real symmetrix matrix/covariance matrix.
  eig_val,eig_vec = np.linalg.eigh(np.cov(cov_mat))


  #The maximum number of eigenvectors obtained along the column i.e 256.
  p = np.size(eig_vec,axis=0)


  #Sort eigenvalues in default ascending order. 
  #The eigenvectors with lower eigenvalues give least info about distribution of data. 
  #np.argsort returns an array of indices of the same shape.
  i = np.argsort(eig_val)

  #This line reverse the order of eigenvalues and sorts and arranges it in descending order. 
  #Eigenvalues are an array with 1 dimension.
  i = i[::-1]           
  eig_val = eig_val[i]           

  #Sorting the eigenvectors in a similar manner. It implies we are selecting a column. Eigenvectors have 2 dimension.
  eig_vec = eig_vec[:,i]

 
  #Sorting the eigenvectors and eigenvalues helps here. Highest ones are given preference.
  #This helps us decide which eigenvectors to choose from the order in which they were sorted. 
  if vals <p or vals >0:
      eig_vec = eig_vec[:,range(vals)]
  
  #Gives us the dot product of the transpose of the eigenvector and the covariance matrix
  score = np.dot(eig_vec.T, cov_mat)

  #Here, Reconstruction is done to make the quality of the images better.
  #Result we get is the data reduced to lower dimensions from higher dimensions.
  recon = np.dot(eig_vec,score)+ np.mean(imgmat, axis=1).T
  return recon

# Passing the image and number of components we are using.
# Displaying 4 pictures alongside with different number of components.
image1 = pca(img,150)
image2 = pca(img,100)
image3 = pca(img,50)
image4 = pca(img,10)

fig,[ax1,ax2,ax3,ax4,ax5] = plt.subplots(1,5)
ax1.axis('off')
ax1.imshow(img,cmap=plt.get_cmap('gray'))
ax1.set_title('True Image')
ax2.axis('off')
ax2.imshow(image1,cmap=plt.get_cmap('gray'))
ax2.set_title('150 PCAs')
ax3.axis('off')
ax3.imshow(image2,cmap=plt.get_cmap('gray'))
ax3.set_title('100 PCAs')
ax4.axis('off')
ax4.imshow(image3,cmap=plt.get_cmap('gray'))
ax4.set_title('50 PCAs')
ax5.axis('off')
ax5.imshow(image4,cmap=plt.get_cmap('gray'))
ax5.set_title('20 PCAs')
plt.show()
