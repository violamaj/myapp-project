import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pylab import rcParams
import seaborn as sb
import scipy
from scipy.stats import spearmanr

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import sklearn.metrics as sm

%matplotlib inline
rcParams["figure.figsize"] =10,5
sb.set_style("whitegrid")

address = "myapp-project/Bankruptcy.csv"
Bankruptcy = pd.read_csv(address)
Bankruptcy.columns = ['ID','failed/healthy','Failed_YR','Cash/Assets']
data = Bankruptcy.iloc[:,[12,4]].values
data_names = ["Failed_YR","Cash/Assets"]
y = cars.iloc[:,[2]].values

Failed_YR = Bankruptcy["Failed_YR"]
Cash/Assets = Bankruptcy["Cash/Assets"]

#Find the Spearmen Coffivient.
spearmanr_coff, p_value = spearmanr(Failed_YR,Cash/Assets)
spearmanr_coff
#negative no correlation

cars.isnull().sum()

sb.countplot(x="Failed_YR",data=Bankruptcy,palette="hls")

x = scale(data)
LogReg = LogisticRegression()

#fit the model
LogReg.fit(x,y)

#print the score
print(LogReg.score(x,y))

#classification report
y_predict = LogReg.predict(x)
from sklearn.metrics import classification_report
report = classification_report(y,y_predict)
print(report)
