import numpy as np
import numpy.random as nr
import scipy.stats as sts
import matplotlib.pyplot as plt
import pandas as pd
from fitter import Fitter, get_common_distributions, get_distributions
import seaborn as sns; sns.set()
from pylab import plot, show, axis, subplot, xlabel, ylabel, grid

from Distributions_Fitter import DistributionFitter
from Distributions_Fitter import MultivariateFitter