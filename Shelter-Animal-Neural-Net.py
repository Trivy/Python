import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Modules for Neural Network
import lasagne
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
#from nolearn.lasagne import objective

# # Modules to visualize Neural Network
# from nolearn.lasagne.visualize import draw_to_notebook
# from nolearn.lasagne.visualize import plot_loss
# from nolearn.lasagne.visualize import plot_conv_weights
# from nolearn.lasagne.visualize import plot_conv_activity
# from nolearn.lasagne.visualize import plot_occlusion
# from nolearn.lasagne.visualize import plot_saliency


# Paths for csv data

#FTRAIN = '~/Documents/Big-data/Shelter-Animal/train.csv'
#FTRAIN = 'C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Kaggle/Shelter-Animal/train.csv'
FTRAIN = 'train.csv'

#FTEST = '~/Documents/Big-data/Shelter-Animal/test.csv'
#FTEST = 'C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Kaggle/Shelter-Animal/test.csv'
FTEST = 'test.csv'

#FEXPORT = 'C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Kaggle/Shelter-Animal/Export-check.csv'
FEXPORT = 'Export-check.csv'

# Introduce the different possible outcomes
Outcomes=[
    'Return_to_owner',
    'Euthanasia',
    'Adoption',
    'Transfer',
    'Died']

Nb_Outcomes=len(Outcomes)

# Introduce names for the features
listFeat=[
    'feat1',
    'feat2',
    'feat3',
    'feat4',
    ]

Nb_listFeat=len(listFeat)

def GenLabels(df):
    df['Label']=\
    df['Adoption']\
    +2*df['Died']\
    +3*df['Euthanasia']\
    +4*df['Return_to_owner']
    +5*df['Transfer']
    return df
# Careful! the order of the labels matters!

def preprocess(df, OptArg=False):
    # The optional argument 'OptArg' indicates whether we should evaluate the outcome columns (train case)
    
    
    # ==============================================
    # Preprocessing: 'Named', 'IsCat', 'AgeNb'
    # ==============================================
    
    # Create and evaluate the column 'HasName'
    df['Named']=(df['Name'].notnull())
    
    # Create and evaluate the column 'IsCat'
    df['IsCat']=(df['AnimalType']=='Cat')
    
    # Express all 'AgeuponOutcome' in days in 'AgeNb'
    #
    # From Andy's script (https://www.kaggle.com/andraszsom/shelter-animal-outcomes/age-gender-breed-and-name-vs-outcome),
    # slightly modified: if the entry is "Nan", returns Nan.
    def age_to_days(item):
        # convert item to list if it is one string
        if type(item) is str:
            item = [item]
        ages_in_days = np.zeros(len(item))
        for i in range(len(item)):
            # check if item[i] is str
            if type(item[i]) is str:
                if 'day' in item[i]:
                    ages_in_days[i] = int(item[i].split(' ')[0])
                if 'week' in item[i]:
                    ages_in_days[i] = int(item[i].split(' ')[0])*7
                if 'month' in item[i]:
                    ages_in_days[i] = int(item[i].split(' ')[0])*30
                if 'year' in item[i]:
                    ages_in_days[i] = int(item[i].split(' ')[0])*365    
            else:
                # item[i] is not a string but a nan
                ages_in_days[i] = np.nan
        return ages_in_days
    
    df['AgeNb']=np.maximum(2,age_to_days(df['AgeuponOutcome']))
    # Those with 0 age are send to 2 days (there are many animals with 'AgeNb' <=10)
        
    # sum(df['AgeNb'].isnull()) yields 18
    # Replace by the average age:
    mean_age=df['AgeNb'].mean()
    
    df['AgeNb'].fillna(mean_age,inplace=True)
    
    for i in range(Nb_listFeat):
        df[listFeat[i]]=np.log(df['AgeNb'])**(i/2)
    
    # Create the different Outcome columns
    for i in range(len(Outcomes)):
        df[Outcomes[i]]=0
    
    # ==================================================
    # Preprocessing 'Sex'
    # ==================================================
    # Extract "male/female" and 'Intact/not" from gender
    
    # Fill in missing 'SexuponOutcome' with 'Unknown'
    df['SexuponOutcome'].fillna('Unknown',inplace=True)
    
    # Create column 'IsIntact'
    df['IsIntact']=(df['SexuponOutcome'].str.contains('Intact'))
    
    # Create column 'IsFemale'
    df['IsFemale']=(df['SexuponOutcome'].str.contains('Female'))
    
    # NB: implicitly, if df['SexuponOutcome']=='Unknown', then the case is treated as "Neutered Male" (as this is the most common designation among the 4 possible cases).
    
    if OptArg :
        # Evaluate the different Outcome columns
        for i in xrange(len(Outcomes)):
            dftrain[Outcomes[i]]=(dftrain['OutcomeType']==Outcomes[i])
        #df=GenLabels(df)
    return df

# ==================================================
# Neural network model: using previous (crude) preprocessing
# ==================================================

# Introduce the columns we want to use for the Neural network
listNN=[
    'Named',
    'IsCat',
    'IsIntact',
    'IsFemale',
    'AgeNb'
    ]

listNN=listNN+listFeat

Nb_features=len(listNN)

def extract_X(df):
    # Returns X (extracted features) 
    X=df.ix[:,listNN].astype(np.float32).values
    return X

def extract_y(df):
    # Returns y (extracted features) 
    y=df.ix[:,Outcomes].values
    #N_y=len(y)
    #np.array(y).astype(np.float32).reshape((N_y,1))
    return np.array(y).astype(np.float32)


# Neural Network properly speaking:
#
net1 = NeuralNet(
    layers=[
        ('input', lasagne.layers.InputLayer),
        ('hidden1', lasagne.layers.DenseLayer),
        ('hidden2', lasagne.layers.DenseLayer),
        ('output', lasagne.layers.DenseLayer),
        ],
    # three layers: one hidden layer
    # layer parameters:
    input_shape=(None, Nb_features),  # Nb_features input per batch
    hidden1_num_units=100,  # number of units in hidden1 layer
    hidden2_num_units=100,  # number of units in hidden2 layer
    output_nonlinearity=lasagne.nonlinearities.softmax,  # output layer uses softmax function
    output_num_units=Nb_Outcomes,  # Nb_Outcomes target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.001, # Initial value: 0.01
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=50,  # we want to train this many epochs
    verbose=1,
    )

# ==================================================
# Execution
# ==================================================

# Training section:
dftrain = pd.read_csv(FTRAIN,header=0)
# reads the train file and integrate it into a Pandas dataframe.
# sep=';' to use only ";" as delimiters
# decimal=',' to use "," as decimal point (and not "." (default behavior))

preprocess(dftrain,True)

X_train=extract_X(dftrain)

y_train=extract_y(dftrain)

net1.fit(X_train,y_train)


# Prediction section:
#
# Read test file
dftest = pd.read_csv(FTEST,header=0)

preprocess(dftest)

X_test=extract_X(dftest)

y_predict=net1.predict(X_test)
# 
# SimpleModelPredict(dftest)
# 
# # =================================================
# # Export of results in expected submission format
# # =================================================
#         
# listExport=[\
#     'ID',
#     'Name',
#     'Adoption',
#     'Died',
#     'Euthanasia',
#     'Return_to_owner',
#     'Transfer'
# ]
# 
# dExport=dftest.ix[:,listExport]
# 
# dExport.to_csv(FEXPORT, index=False)
# 
# # =================================================
# # Conversion in Numpy array
# # =================================================
# 
# # # Convert to a Numpy array, getting ride of first column
# # Export=dExport.ix[:,1:6].values
# 
# # =================================================
# # Blah
# # =================================================
# 
# def Nb_in_classes(dftest):
#     Nb_class=np.zeros((2,2,2,2))
#     # Indices:
#     # 1/ 'Named' (or not) (x2)
#     # 2/ 'IsCat' (x2)
#     # 3/ 'IsIntact' (x2)
#     # 4/ 'IsFemale' (x2)
#     for iNamed in xrange(2):
#         for iIsCat in xrange(2):
#             for iIsIntact in xrange(2):
#                 for iIsFemale in xrange(2):
#                     Nb_class[iNamed,iIsCat,iIsIntact,iIsFemale]= sum((dftest['Named']==iNamed) & (dftest['IsCat']==iIsCat) & (dftest['IsIntact']==iIsIntact) & (dftest['IsFemale']==iIsFemale))
#     return Nb_class
# 
# Nb_class=Nb_in_classes(dftest)
# 
# np.min(Nb_class)
# # returns 191 (for Nb_class[0,1,0,1])
# # There are 5 outcomes, so at least 38 points available for each variables.
