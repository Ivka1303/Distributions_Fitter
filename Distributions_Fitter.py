import numpy as np
import numpy.random as nr
import scipy.stats as sts
import matplotlib.pyplot as plt
import pandas as pd
from fitter import Fitter, get_common_distributions, get_distributions
import seaborn as sns; sns.set()
from pylab import plot, show, axis, subplot, xlabel, ylabel, grid

class DistributionFitter:
    
    def __init__(self, data):
        '''
        Initializes the core attributes of the class:
            quant: bool
                boolean that identifies whether the input datapoints are quantitative or not (a.k.a. qualitative) 
            data: list
                lists of values that correspond to a given vector
        
        Parameters:
        -----------
        data: list
            vector values
        '''
    
        self.data = data
        
        #checkign whether the input vector includes quanlitative data
        checker = [True for char in data if type(char) == int or type(char) == float]
        if len(checker) != len(data) and len(checker) != 0:
            raise TypeError('Datapoints in this vector are both quantitative and qualitative. \nPlease, check this and make sure all the datapoints are of the same type.')
        elif len(checker) == 0:
            self.quant = False
        else:
            self.quant = True
        
        
    def Prob_Of_Models(self, models, priors = False):
        '''
        Calculates the posterior probabiliites of given models being the true models for the data given
        
        Parameters:
        -----------
        models: list
            list of models we want to check for the fit in a scipy.stats (sts.{model name}) format
        priors: list or bool
            list of prior probabilities if there are such ones. 
            Otherwise, False, assuming all the models have the same prior probabilities
        
        Returns:
        --------
        Printed out posterior probabilities of the models given 
        '''
        
        n = len(models)

        #If prior probabilities weren't given assigning equal priors for every model
        if not priors:
            priors = []
            for model in range(len(models)):
                priors.append(1/len(models))

        #Creating a list of likelihoods for every model given the data
        likelihoods = []
        for model in range(len(models)):
            likelihoods.append(np.prod(models[model].pdf(self.data)))

        #Finding the denominator in the Bayes Theorem formula 
        ### aka summing up all the likelihoods multiplied by the corresponding priors
        to_sum = []
        for model in range(len(models)):
            to_sum.append(likelihoods[model]*priors[model])
        denominator = sum(to_sum)

        #denominator = sum(likelihood*prior for prior in priors for likelihood in likelihoods if likelihoods.index(likelihood) == priors.index(prior))
        ##fancier way ^ but likelihood didn't get updated here for a corresponding model :') for whatever reason...

        #Finding the posterior probabilities using the Bayes Formula
        posteriors = []
        for model in range(len(models)):
            posteriors.append(round(likelihoods[model]*priors[model]/denominator, 3))

        #Creating the sictionary to match posteriors and corresponding models
        lib = {}
        for model in range(len(models)):
            lib[posteriors[model]] = models[model]
            
        #Sorting the dictionary by the posterior probabilities, so the most likely models are displyed first
        posteriors[:] = sorted(posteriors)[::-1]
        
        print('Posterior probabilities of the given models explaining the data:')

        #Printing out the probabilities for the models
        for posterior in range(len(posteriors)):
              print(f"Model {models.index(lib[posteriors[posterior]])+1} = {posteriors[posterior]}")
                
            
    def Models_Plots(self, models):
        '''
        Plots the models given as well as the data to visualize the distributions
        
        Parameters:
        -----------
        models: list
            list of scipy.stats models we want to check/compare in a scipy.stats (sts.{model name}) format 
        
        Returns:
        One plot with the models and data given
        '''
        
        #Setting up the plot
        plt.figure()

        #Finding the range of x-values to look at
        x = np.linspace(min(self.data)-(len(self.data)/4), max(self.data)+(len(self.data)/4))

        #Creating plots for every model given
        for model in models:
            plt.plot(x, model.pdf(x), 
                     label=f'Model{models.index(model)+1}')

        #Plotting the data given in a histogram form
        plt.hist(self.data, edgecolor='white', 
                 density=True, color='gray', alpha=0.333, label='data')
        
        #Displying all the plots along with the legend
        plt.legend()
        plt.show()  
        
        
    def Models_Fitter_Helper(self, quantity = 'Common'):
        '''
        Finds 5 most likely models from all the models we can possibly consider using the scipy library
        
        Parameters:
        -----------
        quantity: str
            the quantity of models we want to analyze. 'All' corresponds to all the scipy models available
            while 'Common' takes only the most popular scipy models used
        
        
        Returns:
        --------
        f: list
            parameters list of the fitted distributions
        Y: list
            list of 5 most suitable models available and their corresponding parameters in a printed out format
        '''
        
        if quantity == "All":
            #Taking all distributions available on scipy to compare
            f = Fitter(self.data,
               distributions= get_distributions())
        elif quantity == "Common":
            #Taking all the popular distributions to compare
            f = Fitter(self.data,
               distributions= get_common_distributions())
            

        #Fitting them to the data given and calculating posteriors
        f.fit()
        
        #Creating a dataframe with the distributions and probabilites
        df = f.summary()
        
        #Deleting the index column
        df.reset_index(inplace=True)
        
        #Taking solely the fisrt column with the names of the models
        first_column = df.iloc[:,0]
        
        #Converting that column to a list
        Y = first_column.tolist()
        
        return f
    
        
    def Models_Fitter(self, quantity = 'Common'):
        '''
        Finds 5 most likely models for the quantitative data 
        or multinomial distribution if the input data is qualitative
        
        Parameters:
        -----------
        quantity: str
            the quantity of models we want to analyze. 'All' corresponds to all the scipy models available
            while 'Common' takes only the most popular scipy models used
        
        
        Returns:
        --------
        If input data is quantitative:
            Printed out list of 5 most suitable models available and their corresponding parameters 
        If the data is qualitative:
            Printed out multinomoial distribution with the fitted parameters
        '''
        
        if self.quant == True:
            #finding the most likely models and their respective parameters
            f, Y = self.Models_Fitter_Helper(quantity)

            #Printing out the most likely models
            print("Five the most likely distributions:")
            for dist in Y:
                print(dist, f.fitted_param[dist])
            
        else:
            #getting unique qualitative datapoints
            ##reversing the list, so that when we append values to a dictionary later 
            ##the probabilities are in the same order as the input data
            values = reversed(list(set(self.data)))
            probs = {}
            
            for value in values:
                #defining the probability of a value by calculating its frequency in a given vector
                probs[value] = round(self.data.count(value)/len(self.data), 2)
            print(f'The input vector follows a multinomial distribution with the following probability parameters: \n{probs}')
            
        
    def Display_All_Models(self):
        '''
        Displays all the distributions scipy library can analyze   
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        list of the distributions
        '''
        
        return get_distributions()
    
        
    def Display_Common_Distributions(self):
        '''
        Displays the most popular distributions scipy library can analyze        
        
        Parameters:
        -----------
        None
    
        Returns:
        --------
        list of the distributions
        '''
        return get_common_distributions()
    
    
    def Sampler(self, N, model = 0):
        '''
        Creates a new sample that would follow the same distribution as the input vector
        
        Parameters:
        -----------
        N: int
            the size of the sample we want to get
        model: scipy.stats distribution
            a distribution we want to sample from for the quantitative data 
            (can be easily found using the Models_Fitter method above)
            Set to zero by default in case the input data we want to create a sample for is qulitative
        
        Returns:
        --------
        A list of sampled datapoints
        
        '''
        
        if self.quant == True:
            if not model:
                raise TypeError('Sampler() missing 1 required positional argument: model. \nThe model must be given for the analysis as the input data is quantitave')
            else:
                #randomly sampling N datapoints from a distribution given
                return model.rvs(N)
            
        else:
            #finding the probabilities of the qualitative values given
            values = reversed(list(set(self.data))) #so the output is in the same order as the input
            print(values)
            probs = {}
            
            for value in values:
                probs[value] = round(self.data.count(value)/len(self.data), 2)
              
            #randomly sampling from a distribution with the same number of unique values and their probabilities
            model = nr.multinomial(N, list(probs.values()))
            
            return [list(values)[idx] for idx in range(len(values)) for repetition in range(model[idx])] 
            

    def Sampler_Plotter(self, N, model = 0):
        '''
        Displays a dot plot for quantitave sample and a histogram for qualitative data
        
        Parameters:
        -----------
        N: int
            the size of the sample
        model: scipy.stats distribution
            a distribution we want to sample from for the quantitative data 
        
        Returns:
        --------
        A plot of the data sasmpled
        '''
        
        #Setting up the plot
        plt.figure()
                  
        #sampling data of a given size and distribution if any 
        sample = self.Sampler(N, model)
        
        if self.quant == True:
            plt.plot(sample, 'b.')
            
        else:
            plt.hist(sample)
            
            
class MultivariateFitter:
    
    def __init__(self, input_data):
        '''
        Initializes the core attributes of the clfass
        
        Parameters:
        -----------
        vectors: list
            lists of lists of values that correspond to given vectors
        '''
        
        #if the user enters a name of a csv file
        if type(input_data) == str:
            df = pd.read_csv(input_data)
            data = []

            for column in df.columns:

                # Storing the rows of a column
                # into a temporary list
                li = df[column].tolist()

                # appending the temporary list
                data.append(li)

            #storing the names of the vectors separately
            self.titles = list(df.columns.values)
        
        #if the user directly inputs vectors
        else:
            data = input_data
            self.titles = input('Please, enter the names of vectors in the order of the vectors given. \nLeave one space among the names and do not use commas.').split()
            if len(self.titles) != len(data):
                raise TypeError('The number of vector labels is different from the number of vectors present')
       
        #checking if all the input data is quantitave. Otherwise return an error
        for column in data:
            curr = [True for char in column if type(char) == int]
            if len(curr) != len(column) and len(curr) != 0:
                raise TypeError(f'Some of the input data in vector {data.index(column)+1} is not quantitave. \nPlease, check this and make sure all the datapoints are integers/floats.')
            elif len(curr) == 0:
                raise TypeError(f'All the datapoints in vector {data.index(column)+1} are qualitave. \nPlease, check this and make sure all the data points are integers/floats.')
    
        #converting given vectors into arrays 
        self.data = np.array(data)
        
        
    def Sampler(self, N):
        '''
        Samples N values for every given vector simulated
        
        Parameters:
        -----------
        N: int
            number of values to sample
        
        Returns:
        --------
        x: dict
            Dictionary of the vectors sampled where keys correspond to the vector labels      
        
        '''
        
        #Finding the covariance matrix for the given vectors
        K_0 = self.CovMatrix_GivenVectors()
        
        #finding the standard deviations in the vectors and finding the mean one
        sd = sum([np.std(vector) for vector in self.data])/len(self.data)
        
        # Eigenvalues covariance function.
        np.linalg.eigvals(K_0)

        # Define epsilon.
        epsilon = 0.0001
        
        #Define dimension
        d = len(self.data)

        # Add small pertturbation. 
        K = K_0 + epsilon*np.identity(d)
        
        #  Cholesky decomposition.
        L = np.linalg.cholesky(K)
        np.dot(L, np.transpose(L))

        # Number of samples. 
        n = N

        #Randomly sampling from a normal distribution with the average SD based on the given vectors
        u = np.random.normal(loc=0, scale=sd, size=d*n).reshape(d, n)
        
        #set mean vector
        pre_m = [np.mean(vector) for vector in self.data]      
        m = np.array(pre_m).reshape(len(self.data), 1)
        
        #Relocating the sampled data based on the mean vector 
        x = m + np.dot(L, u) 
        
        #creating a dictionary to keep track of the vectors samples and their names
        with_names = {}
        for i in range(len(x)):
            with_names[self.titles[i]] = x[i]
        
        #returning the samples
        return with_names
    
    
    def CovMatrix_GivenVectors(self):
        '''
        Finds the covariance matrix for the vectors given
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        K_0: the covariance matrix
        
        '''
                
        #Findnig the coveriance matrix        
        K_0 = np.cov(self.data, bias = True)
        
        return K_0
    
    
    def CovMatrix_Sampled(self, N):
        '''
        Finds the covariance matrix for the vectors sampled in the simulation
        
        Parameters:
        -----------
        N: int
            the number of values you want to sample
        
        Returns:
        --------
        K_1: list
            the covariance matrix
        '''
        
        #Creates the sample in case it was not created earlier
        x = self.Sampler(N)
        #Calculating the covariance matrix
        K_1 = np.cov(x,bias=True)
        
        return K_1
    
    
    def Sample_Plotter(self, N):
        '''
        Creates scatter plots for the vectors sampled
        
        Parameters:
        -----------
        N: int
            the number of values you want to sample

        Returns:
        --------
        The list of plots
        '''
        
        #Sampling N values for each given vector from the simulation in case it wasn't done previously
        x = self.Sampler(N)
        
        #Keeping track of the plots created
        count = 0     
        
        #Creating scatter plots for every possible combination of two vectors
        for idx in range(len(x)):
            for idx2 in range(idx+1, len(x)):
                count += 1
                globals()['plot%s' % count] = plt.figure(count)
                plt.plot(list(x.values())[idx], list(x.values())[idx2], 'b.')
                xlabel(f'{list(x.keys())[idx]}')
                ylabel(f'{list(x.keys())[idx2]}')
                
        #Displaying all the plots created
        plt.show()
        
        
    def Given_Plotter(self):
        '''
        Creates scatter plots for the vectors given
        
        Parameters:
        -----------
        None

        Returns:
        --------
        The list of plots
        '''
    
        #Keeping track of the plots created
        count = 0    
        
        #Creating scatter plots for every possible combination of two vectors
        for idx in range(len(self.data)):
            for idx2 in range(idx+1, len(self.data)):
                count += 1
                globals()['plot%s' % count] = plt.figure(count)
                plt.plot(self.data[idx], self.data[idx2], 'b.')
                xlabel(f'{self.titles[idx]}')
                ylabel(f'{self.titles[idx2]}')
        
        #Displaying all the plots created
        plt.show()
