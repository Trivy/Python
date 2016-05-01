import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#from lasagne import layers
#from lasagne.updates import nesterov_momentum
#from nolearn.lasagne import NeuralNet

#FTRAIN = '~/Documents/Big-data/Shelter-Animal/train.csv'
FTRAIN = 'C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Kaggle/Shelter-Animal/train.csv'

#FTEST = '~/Documents/Big-data/Shelter-Animal/test.csv'
FTEST = 'C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Kaggle/Shelter-Animal/test.csv'

FEXPORT = 'C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Kaggle/Shelter-Animal/Export-check.csv'


# Introduce the different possible outcomes
Outcomes=[
    'Return_to_owner',
    'Euthanasia',
    'Adoption',
    'Transfer',
    'Died']


def preprocess(df):
    
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
    
    df['AgeNb']=age_to_days(df['AgeuponOutcome'])
    
    # sum(df['AgeNb'].isnull()) yields 18
    # Replace by the average age:
    mean_age=df['AgeNb'].mean()
    
    df['AgeNb'].fillna(mean_age,inplace=True)
    
    # Create the different Outcome columns
    for i in xrange(len(Outcomes)):
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

# ==================================================
# Simple model: divide animals according to our previous booleans
# ==================================================

def SimpleModelFit(dftrain):
    # ================================================
    # Prepare outcomes
    # ================================================
    
    # Evaluate the different Outcome columns
    for i in xrange(len(Outcomes)):
        dftrain[Outcomes[i]]=(dftrain['OutcomeType']==Outcomes[i])
    

    #
    # Initialize the table
    probability_table=np.zeros((5,2,2,2,2))
    # Indices:
    # 1/ possible outcomes (x5)
    # 2/ 'Named' (or not) (x2)
    # 3/ 'IsCat' (x2)
    # 4/ 'IsIntact' (x2)
    # 5/ 'IsFemale' (x2)
    
    for iNamed in xrange(2):
        for iIsCat in xrange(2):
            for iIsIntact in xrange(2):
                for iIsFemale in xrange(2):
                    NbSeries= sum( (dftrain['Named']==iNamed) & (dftrain['IsCat']==iIsCat) & (dftrain['IsIntact']==iIsIntact) & (dftrain['IsFemale']==iIsFemale) )
                    if (NbSeries != 0):
                        for i in xrange(len(Outcomes)):
                            probability_table[i,iNamed,iIsCat,iIsIntact,iIsFemale]=float(sum( (dftrain[Outcomes[i]]) & (dftrain['Named']==iNamed) & (dftrain['IsCat']==iIsCat) & (dftrain['IsIntact']==iIsIntact) & (dftrain['IsFemale']==iIsFemale) ))/ NbSeries
                    else :
                        for i in xrange(len(Outcomes)):
                            probability_table[i,iNamed,iIsCat,iIsIntact,iIsFemale]=np.nan
    return probability_table

# ==================================================
# Simple model: prediction of outcome
# ==================================================
def SimpleModelPredict(df):
    # Loop over all possible values of 'Named', 'IsCat', 'IsIntact' and 'IsFemale'...
    for iNamed in xrange(2):
        for iIsCat in xrange(2):
            for iIsIntact in xrange(2):
                for iIsFemale in xrange(2):
                    # then make a mask, that we are going to use to treat all relevant lines simultaneously
                    test=( (df['Named']==iNamed) & (df['IsCat']==iIsCat) & (df['IsIntact']==iIsIntact) & (df['IsFemale']==iIsFemale) )
                    for i in xrange(len(Outcomes)):
                        df.ix[test,Outcomes[i]]=probability_table[i,iNamed, iIsCat, iIsIntact, iIsFemale]
    return df

# ==================================================
# Execution
# ==================================================

dftrain = pd.read_csv(FTRAIN,header=0)
# reads the train file and integrate it into a Pandas dataframe.
# sep=';' to use only ";" as delimiters
# decimal=',' to use "," as decimal point (and not "." (default behavior))

preprocess(dftrain)

probability_table=SimpleModelFit(dftrain)

# Read test file
dftest = pd.read_csv(FTEST,header=0)

preprocess(dftest)

SimpleModelPredict(dftest)

# =================================================
# Export of results in expected submission format
# =================================================
        
listExport=[\
    'ID',
    'Name',
    'Adoption',
    'Died',
    'Euthanasia',
    'Return_to_owner',
    'Transfer'
]

dExport=dftest.ix[:,listExport]

dExport.to_csv(FEXPORT, index=False)

# =================================================
# Conversion in Numpy array
# =================================================

# # Convert to a Numpy array, getting ride of first column
# Export=dExport.ix[:,1:6].values

# =================================================
# Blah
# =================================================

def Nb_in_classes(dftest):
    Nb_class=np.zeros((2,2,2,2))
    # Indices:
    # 1/ 'Named' (or not) (x2)
    # 2/ 'IsCat' (x2)
    # 3/ 'IsIntact' (x2)
    # 4/ 'IsFemale' (x2)
    for iNamed in xrange(2):
        for iIsCat in xrange(2):
            for iIsIntact in xrange(2):
                for iIsFemale in xrange(2):
                    Nb_class[iNamed,iIsCat,iIsIntact,iIsFemale]= sum((dftest['Named']==iNamed) & (dftest['IsCat']==iIsCat) & (dftest['IsIntact']==iIsIntact) & (dftest['IsFemale']==iIsFemale))
    return Nb_class

Nb_class=Nb_in_classes(dftest)

np.min(Nb_class)
# returns 191 (for Nb_class[0,1,0,1])
# There are 5 outcomes, so at least 38 points available for each variables.
