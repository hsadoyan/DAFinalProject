
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import bokeh
from bokeh.plotting import figure, output_file, show, ColumnDataSource
# import bokeh.charts.utils
import bokeh.io
import bokeh.models
from bokeh.models import HoverTool
import bokeh.palettes
import bokeh.plotting
import random
from random import sample
from sklearn import svm, neighbors
from sklearn.model_selection import ShuffleSplit
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsRegressor


import multiprocessing as mp


# In[2]:


# Preprocessing, normalization done in excel (Xnew = (X-mean)/std))
# If we want to normalize in Python we can use preprocessing.scale()
Data = pd.read_csv('songs4.csv')
Data = Data.iloc[:, 0:18]
#Data = Data.drop(Data[(Data.time_signature > 5)].index)
Data.head()


# In[3]:


# check number of rows
Data.count()


# In[4]:


# Divide into testing and training
x = Data.drop('valence', 1)
y = Data.valence
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20)
train = xtrain
train['valence'] = ytrain
# train = train [0: 5000]
train.head()


# In[5]:


# Create correlation matrix
M = train.corr()


# In[6]:


M


# In[7]:


# correlation matrix
plt.imshow(M)
plt.colorbar()
plt.show()


# In[8]:


# SVD using numpy function
U, E, VT = np.linalg.svd(M)


# In[9]:


plt.plot(E)
plt.show()


# In[10]:


P = np.dot(U[:,:2],np.diag(E[:2]))


# In[11]:


plt.plot(P[:,0], P[:,1],'o')
plt.show()


# In[12]:


N = train.T
N.columns = N.iloc[0]
N = N.drop('ID')
N = N.astype(float)


# In[13]:


# Takes 10-15 minutes with all of the data
N = N.corr()
N.head()


# In[14]:


# Identifies how different songs correlate to each other, there is a fair amount of uniqueness among songs
plt.imshow(N)
plt.colorbar()
plt.show()


# In[15]:


U, E, VT = np.linalg.svd(N)


# In[16]:


# Most of the variance can be explained using the first 8 or so components
plt.plot(E[:10])
plt.show()


# In[17]:


P = np.dot(U[:,:2],np.diag(E[:2]))
print(P)


# In[18]:


# plot first two principal components to get an idea of the shape of the data
_tools_to_show = 'box_zoom,pan,save,hover,reset,tap,wheel_zoom'        
p = figure(plot_width=400, plot_height=400, title=None, tools=_tools_to_show)

# add a circle renderer with a size, color, and alpha
p.circle(P[:,0], P[:,1], size=4, color="navy", alpha=0.2)

# show the results
show(p)


# In[19]:


# SVM on PCA results


# In[20]:


# Further divide training and testing based on principal components
# Slice U and E based on the ideal number of principal components
P = np.dot(U,np.diag(E))
P = P[:,:18]
PCA_xtrain, PCA_xtest, PCA_ytrain, PCA_ytest = train_test_split(P, train['valence'], test_size = 0.2, random_state = 0)


# In[21]:


SVM_clf = svm.SVR(kernel='linear')


# In[22]:


SVM_clf.fit(PCA_xtrain, PCA_ytrain)


# In[23]:


# testing error
SVM_test_ypreds = SVM_clf.predict(PCA_xtest)
MSE = np.mean((SVM_test_ypreds - PCA_ytest)**2)
MSE


# In[24]:


SVM_clf.score(PCA_xtest, PCA_ytest)


# In[25]:


SVM_clf.score(PCA_xtrain, PCA_ytrain)


# In[26]:


# training error
SVM_train_ypreds = SVM_clf.predict(PCA_xtrain)
SVM_test_MSE = np.mean((SVM_train_ypreds - PCA_ytrain)**2)
SVM_test_MSE


# In[29]:


# cross validate for values of C and gamma, start by defining the ranges for each
C_range = [2.0, 3.0]

# Cross validate for optimal value of C
def f(i):
    SVM_clf_C = svm.SVR(kernel='linear', C = i)
    SVM_clf_C.fit(PCA_xtrain, PCA_ytrain)
    SVM_test_ypreds_C = SVM_clf_C.predict(PCA_xtest)
    SVM_test_MSE_C = np.mean((SVM_test_ypreds_C - PCA_ytest)**2)
    print(i)
    return SVM_test_MSE_C
    
pool = mp.Pool(processes=8)
C_MSE = pool.map(f, C_range)


# In[28]:


# Plot values of C vs MSE
_tools_to_show = 'box_zoom,pan,save,hover,reset,tap,wheel_zoom'        
p_C_MSE = figure(plot_width=400, plot_height=400, title=None, tools=_tools_to_show)

# add a circle renderer with a size, color, and alpha
p_C_MSE.circle (C_range, C_MSE, size=10, color="green", alpha=0.5)

# show the results
show(p_C_MSE)


# In[ ]:


# Using our optimal value of C, we cross validate to find the optimal value of gamma

def f2(i)
    SVM_clf_gamma = svm.SVR(kernel='rbf', C = 100, gamma = i)
    SVM_clf_gamma.fit(PCA_xtrain, PCA_ytrain)
    SVM_test_ypreds_gamma = SVM_clf_gamma.predict(PCA_xtest)
    SVM_test_MSE_gamma = np.mean((SVM_train_ypreds - PCA_ytest)**2)
    gamma_MSE.append(SVM_test_MSE_gamma)
    
gamma_MSE = pool.map(f2, gamma_range)


# In[ ]:


# Plot various values of gamma vs MSE
_tools_to_show = 'box_zoom,pan,save,hover,resize,reset,tap,wheel_zoom'        
p_gamma_MSE = figure(plot_width=400, plot_height=400, title=None, tools=_tools_to_show)

# add a circle renderer with a size, color, and alpha
p_gamma_MSE.circle (gamma_range, gamma_MSE, size=10, color="orange", alpha=0.5)

# show the results
show(p_gamma_MSE)


# In[ ]:


# Random Forest

RF_clf = RandomForestRegressor()

# specify parameters and distributions to sample from
parameters_rand = {
    "n_estimators": sp_randint(10, 60),
    "bootstrap": [True, False],
}

# run randomized search
# Accuracy should be comparable to grid search, but runs much much faster
n_iter_search = 20
random_search = RandomizedSearchCV(RF_clf, param_distributions=parameters_rand,
                                   n_iter=n_iter_search,
                                   n_jobs=-1)

random_search.fit(PCA_xtrain, PCA_ytrain)

predicted = random_search.predict(PCA_xtest)

print("PCA with random forest")
random_search.score(PCA_xtest, PCA_ytest)


# In[ ]:


# Lasso (on it's own)
from sklearn import linear_model

# Train
lasso_models = {} # Keyed by alpha
xtrain_no_id = xtrain.iloc[:, 1:]
xtest_no_id  = xtest.iloc[:, 1:]

for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0]:
    lasso_model = linear_model.Lasso(alpha=alpha)
    lasso_model.fit(xtrain_no_id, ytrain)
    
    # Training error
    lasso_train_ypreds = lasso_model.predict(xtrain_no_id)
    lasso_train_MSE = np.mean((lasso_train_ypreds - ytrain) ** 2)
    
    # Testing error
    lasso_test_ypreds = lasso_model.predict(xtest_no_id)
    lasso_test_MSE = np.mean((lasso_test_ypreds - ytest)**2)
    
    # Output
    print("alpha: {}".format(alpha))
    print("training error: {}".format(lasso_train_MSE))
    print("testing  error: {}".format(lasso_test_MSE))
    
    # Save
    lasso_models[alpha] = lasso_model


# In[ ]:


# KNN


# In[ ]:


jig = neighbors.KNeighborsRegressor()


# In[ ]:


jig.fit(PCA_xtrain, PCA_ytrain)


# In[ ]:


knn_test_ypreds = jig.predict(PCA_xtest)


# In[ ]:


knn_test_MSE = np.mean((y_pred2 - y_test)**2)


# In[ ]:


knn_test_MSE


# In[ ]:


jig.score(PCA_xtest, PCA_ytest)


# In[ ]:


n_samples = PCA_xtrain.data.shape[0]


# In[ ]:


n_samples


# In[ ]:


# cross validation for KNN  
kf = KFold(n_samples, n_folds=5, shuffle=False)
print(kf)


# In[ ]:


# we use cross validation to find the optimal number of k
k  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
knn_test_MSE_k = []
for i in k: 
    knn = KNeighborsRegressor(n_neighbors=i)
    scores = cross_val_score(knn, PCA_xtrain, PCA_ytrain, cv=5, scoring='neg_mean_squared_error')
    MSE_k = abs(sum(scores))/5
    knn_test_MSE_k.append(MSE_k)


# In[ ]:


# graph number of k vs mse
_tools_to_show = 'box_zoom,pan,save,hover,resize,reset,tap,wheel_zoom'        
p_knn_MSE = figure(plot_width=400, plot_height=400, title=None, tools=_tools_to_show)

# add a circle renderer with a size, color, and alpha
p_knn_MSE.circle (k, knn_test_MSE, size=10, color="red", alpha=0.5)

# show the results
show(p_knn_MSE)


# In[20]:


PCA_xtrain.shape()

