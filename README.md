# Distributions_Fitter
A package with methods written in Python3 for finding the most probable models for given data as well as creating new samples of the data that follow the distribution as close as possible to one of the input data.

--------------------------------------

In a nutshell, the code for MutivariateFitter given uses the numpy python module to find the covariances among the given vectors (if we analyze multivariate data) and create new sampled vectors that would have the same corresponding covariance matrix. As of now, the code can sample new vectors for the multivariate data on;y from normal distributions. The parameters of the normal distributions for each vector are calculated based on the mean and standard deviation of the given vectors. Besides finding the covariance matrices and sampling new vectors of a custom size, the object allows us to plot the vectors given as well as the new ones sampled. 

DistributionsFitter object is designed for finding the distributions that could model a given single vector with the highest probability. It does so by finding the posterior probabilities of different distributions available given the scipy module. The user can both input the models they want to analyze right away or allow the program to look through all the possible distributions on its own. Also, the object uses a fitter module to find the parameters of the distributions that would provide the most likely model for the given data. As well as the MultivariateFitter, DistributionsFitter can plot the data given and the data sampled. 

# Code Structure
In the very beginning I am importing all the Python modules needed for the program to run smoothly. Also, I made sure to include docstrings that explain what a given method does as well as clearly state which input it requires. You can also observe quite a few comments throughout the code that explain the logic behind every algorithm used within the objects. 

# Running it on Jupiter Notebook 
This is the easiest way to play around with the code given, and this is how I was creating and testing it. 

Just copy and paste the code from the Distributions_Fitter.py file.

# A Side Note
If you use this code in any of your projects, I would really appreciate citing me there.

Hope my fitter comes in handy for you :) 

Also, I would like to thank Prof. Hadavand from Minerva University for providing the idea and supporting me throughout this project. 
