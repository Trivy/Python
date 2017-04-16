import pandas as pd
import numpy as np
import pylab as P
from matplotlib import pyplot as plt

INI_PATH = '/Users/gaest/Dropbox/Travail/Big-data/Bi-licence_Franzi/'
# INI_PATH ='E:/Dropbox/Travail/Big-data/Bi-licence_Franzi/'

FINPUT=INI_PATH+'Data_bi-licence/2017/APB-2017-3.csv'
FOUTPUT=INI_PATH+'Data_bi-licence/2017/ExportSortedv0.csv'
FSPECIAL=INI_PATH+'Data_bi-licence/2017/ExportSpecialv0.csv'

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv(FINPUT,sep=';',decimal=',',header=0)
# sep=';' to use only ";" as delimiters
# decimal=',' to use "," as decimal point (and not "." (default behavior))

# Keep only those 
# _ whose application is confirmed ('Candidature validée (Code)'==1)
# _ and whose series are neither
#   * 'STMG', 'Sciences et Technologies du Management et de la Gestion'
#   * 'P', 'Professionnelle'
#   * 'STI2D', "Sciences et Technologies de l'Industrie et du Développement Durable"
#   * 'ST2S', 'Sciences et technologies de la santé et du social'
#   * 'STAV'
#   * 'ST2A'
#   * 'STL'
#   * 'BMA'
#   * 'STI'
#   * 'STG'
dred=df[(df['Candidature validée (Code)']==1) & \
    ((df['Série (Code)']!='STMG') & \
        (df['Série (Code)']!='P') & \
        (df['Série (Code)']!='STI2D') & \
        (df['Série (Code)']!='ST2S') & \
        (df['Série (Code)']!='STAV') & \
        (df['Série (Code)']!='ST2A') & \
        (df['Série (Code)']!='STL') & \
        (df['Série (Code)']!='BMA') & \
        (df['Série (Code)']!='STI') & \
        (df['Série (Code)']!='STG')
    )]
# NB: si on donne une liste de séries admissibles, on ne traite plus le cas des scolarisés à l'étranger, reprise d'études, ...
# NB: it would also be possible to sort according to 'Type de classe (code)' (and keep only those with type != 2)?


## =======================================
# different test cases
##
# Introduce testRegFrance (un des trois cas : "En Terminal" ou "Scolarisé dans le supérieur en France" ou "Non scolarisé")
testRegFrance=\
        (dred['Profil du candidat (Code)']==1) | \
        (dred['Profil du candidat (Code)']==11) | \
        (dred['Profil du candidat (Code)']==12) | \
        (dred['Profil du candidat (Code)']==15)
        
testTerminale=\
    (dred['Profil du candidat (Code)']==1)

testSuperieur=\
    (dred['Profil du candidat (Code)']==11) | \
    (dred['Profil du candidat (Code)']==12)

testNonScol=\
    (dred['Profil du candidat (Code)']==15)

## =======================================
# column 'Bonus'
##

dred.loc[:,'Bonus']=np.nan



## Pretreatment "Terminale"
    
# Estimation of the mean 'Avis du CE (Code)':
# dred['Avis du CE (Code)'].mean()

# Estimation of the mean 'Niveau de la classe':
# dred['Niveau de la classe'].map( {'Faible': -1, 'Assez bon': 0, 'Moyen': 1, 'Bon':2, 'Très bon':3}).mean()

# If 'Avis du CE (Code)' is not available, fill in with 'Favorable'
dred.loc[testTerminale & (dred['Avis du CE (Code)'].isnull()),'Avis du CE (Code)']=3

# If 'Niveau de la classe' is not available, fill in with 'Moyen'
dred.loc[testTerminale & (dred['Niveau de la classe'].isnull()),'Niveau de la classe']='Moyen'

# If 'Option internationale (Code)' is not available, fill in with '0'
dred.loc[testTerminale & (dred['Option internationale (Code)'].isnull()),'Option internationale (Code)']='0'
# [2017] NB: I tried with fillna, but that did not seem to work!

# Compute the overall mark for 'testCond'
def ComputeBonusTale(testCond,dred):
    dred.loc[testCond,'Bonus']=\
    2*(dred.ix[testCond,'Avis du CE (Code)']-1)+\
    dred.ix[testCond,'Niveau de la classe'].map( {'Faible': -1, 'Assez bon': 0, 'Moyen': 1, 'Bon':2, 'Très bon':3})+\
    4*dred.ix[testCond,'Option internationale (Code)'].map( {'0': 0,'B':1,'1': 1, '2': 1})
# NB:
# _ avis du CE : entre -2 et 4 points
# _ Niveau de la classe : entre -1 et 3 points
# _ Option internationale : entre 0 et 4 points 

ComputeBonusTale(testTerminale,dred)

## Pretreatment "Superieur" et "Non scolarisé"

# Possible 'Type établissement (Code)'
# 1 - 'Lycée à classe postbac' (NB: includes CPGE)
# 2 - 'Lycée sans classe postbac'
# 3 - 'I.U.T' (NB [2017], only one case
# 4 - 'Université'

# Bonus [2017] : 
# _ CPGE (Code == 1) : 11 pts,
# _ 'Prépa formation en Sciences Politiques (hors CPGE)' (Code == 212) : 11 pts
# _ 'Cycle universitaire préparatoire aux grandes écoles' (Code == 83) : 11 pts
# _ 'Formation en Sciences Politiques' (Code == 211) (11 pts)
# _ le reste = 4 pts (avis CE = 'Favorable', niveau de la classe = 'Bon', pas d'option internationale)
 

# Compute the overall mark for 'testCond'
def ComputeBonusSup(testCond,dred):
    dred.loc[testCond,'Bonus']=\
    4 +\
    7*(testCond & (dred['Type de formation (Code)']==1))+\
    7*(testCond & (dred['Type de formation (Code)']==212))+\
    7*(testCond & (dred['Type de formation (Code)']==83))+\
    7*(testCond & (dred['Type de formation (Code)']==211))

ComputeBonusSup( (testSuperieur | testNonScol) ,dred)

## =======================================
# Completion of column 'Note Anglais'
## =======================================

# Create new column 'Note Anglais' in dred
dred.loc[:,'Note Anglais']=np.nan

# Create new column 'Moyenne LV 1 Tale' in dred
dred.loc[:,'Moyenne LV 1 Tale']=np.nan

## (testSuperieur | testNonScol)
#
ELV1= "Note à l'épreuve de Langue vivante 1"
ELV2= "Note à l'épreuve de Langue vivante 2"
EL1 = "Note à l'épreuve de Langue 1"
EL2 = "Note à l'épreuve de Langue 2"
# If (testSuperieur | testNonScol) and
# if 'LV1 Bac (Code)'==2 ('Anglais'), then take the mark 'Note Anglais' as mean of "Note à l'épreuve de Langue vivante 1" and "Note à l'épreuve de Langue 1"
# [2017] NB: sur nos données, pas de cas avec les deux simultanément non-nuls !
dred.loc[ (testSuperieur | testNonScol) & (dred['LV1 Bac (Code)']==2), 'Note Anglais'] =\
    dred.loc[ (testSuperieur | testNonScol) & (dred['LV1 Bac (Code)']==2), [ELV1, EL1] ].mean(axis=1)
    
# If (testSuperieur | testNonScol) and
# if 'LV2 Bac (Code)'==2 ('Anglais'), then take the mark 'Note Anglais' as "Note à l'épreuve de Langue vivante 2"
dred.loc[ (testSuperieur | testNonScol) & (dred['LV2 Bac (Code)']==2), 'Note Anglais'] =\
    dred.loc[ (testSuperieur | testNonScol) & (dred['LV2 Bac (Code)']==2), [ELV2, EL2] ].mean(axis=1)

## testTerminale
#
# Fill in column 'Moyenne LV 1 Tale' in dred
MLV1T1='Moyenne candidat en Langue vivante 1 Trimestre 1'
MLV1T2='Moyenne candidat en Langue vivante 1 Trimestre 2'

dred.loc[testTerminale,'Moyenne LV 1 Tale']=\
    dred.loc[testTerminale,[MLV1T1]+[MLV1T2]].mean(axis=1)


# Create new column 'Moyenne LV 2 Tale' in dred
dred.loc[:,'Moyenne LV 2 Tale']=np.nan

# Fill in column 'Moyenne LV 2 Tale' in dred
MLV2T1='Moyenne candidat en Langue vivante 2 Trimestre 1'
MLV2T2='Moyenne candidat en Langue vivante 2 Trimestre 2'

dred.loc[testTerminale,'Moyenne LV 2 Tale']=\
dred.ix[testTerminale,[MLV2T1]+[MLV2T2]].mean(axis=1)

# First run for 'Note Anglais': if 'LV 2 scolarité'=='Anglais', then set 'Note Anglais' = 'Langue vivante 2 (note)'
dred.loc[testTerminale & (dred['LV 2 scolarité']=='Anglais'),'Note Anglais']=dred.ix[testRegFrance & (dred['LV 2 scolarité']=='Anglais'),'Langue vivante 2 (note)']

# Second run for 'Note Anglais': if 'LV 1 scolarité'=='Anglais', then set 'Note Anglais' = 'Langue vivante 1 (note)'
dred.loc[testTerminale & (dred['LV 1 scolarité']=='Anglais'),'Note Anglais']=\
dred.ix[testTerminale & (dred['LV 1 scolarité']=='Anglais'),'Langue vivante 1 (note)']

# [2016] NB: on the available data, there is no case of ( ('LV 2 scolarité'=='Anglais') & ('LV 1 scolarité'=='Anglais') ), but running 'LV 1 scolarité'=='Anglais' in second ensure that these marks would take precedence. 
# # [2017] to check it on data (still outputs 0 lines):
# dred[ (dred['LV 1 scolarité']=='Anglais') & (dred['LV 2 scolarité']=='Anglais')].shape

## for those that remain:

# Next try: fill in those that remain, using the marks from Tale:
# if (dred['LV 1 scolarité']=='Anglais'), then update 'Note Anglais' with 'Moyenne LV 1 Tale'.
dred.loc[testRegFrance & (dred['Note Anglais'].isnull()) & (dred['LV 1 scolarité']=='Anglais'),'Note Anglais']=\
    dred.ix[testRegFrance & (dred['Note Anglais'].isnull()) & (dred['LV 1 scolarité']=='Anglais'),'Moyenne LV 1 Tale']

# if (dred['LV1']=='Anglais'), then update 'Note Anglais' with 'Moyenne LV 1 Tale'.
# [2017] cas des 'scolarisés dans le supérieur en France' (càd 'Profil du candidat (Code)'==11)
dred.loc[testRegFrance & (dred['Note Anglais'].isnull()) & (dred['LV1']=='Anglais'),'Note Anglais']=\
dred.ix[testRegFrance & (dred['Note Anglais'].isnull()) & (dred['LV1']=='Anglais'),'Moyenne LV 1 Tale']

# Last try: fill in those that remain, using the marks from Tale:
# if (dred['LV 2 scolarité']=='Anglais'), then update 'Note Anglais' with 'Moyenne LV 2 Tale'.
dred.loc[testRegFrance & (dred['Note Anglais'].isnull()) & (dred['LV 2 scolarité']=='Anglais'),'Note Anglais']=\
dred.ix[testRegFrance & (dred['Note Anglais'].isnull()) & (dred['LV 2 scolarité']=='Anglais'),'Moyenne LV 2 Tale']

## =======================================
# Completion of column 'Note Histoire'
## =======================================

# Create new column 'Moyenne Histoire/Géographie Tale' in dred
dred.loc[:,'Note Histoire']=np.nan

## (testSuperieur | testNonScol)
#
EHGA= "Note à l'épreuve de Histoire - géographie (épreuve anticipée)"
EHG= "Note à l'épreuve de Histoire-Géographie"
EH= "Note à l'épreuve de Histoire"

# If (testSuperieur | testNonScol) and
# if 'LV1 Bac (Code)'==2 ('Anglais'), then take the mark 'Note Hitoire' as mean of EHGA, EHG and EH
# [2017] NB: sur nos données, pas de cas avec deux des trois simultanément non-nuls !
dred.loc[ (testSuperieur | testNonScol),'Note Histoire'] =\
    dred.loc[ (testSuperieur | testNonScol), [EHGA, EHG, EH] ].mean(axis=1)

## testTerminale

# Use 'Histoire/Géographie (note)' as base for 'Note Histoire' (when available)
dred.loc[testTerminale, 'Note Histoire']=\
    dred.ix[testTerminale, 'Histoire/Géographie (note)']

## for those that remain:

# If necessary and possible, complete with the mean of
MHT1='Moyenne candidat en Histoire/Géographie Trimestre 1'
MHT2='Moyenne candidat en Histoire/Géographie Trimestre 2'
dred.loc[testRegFrance & (dred['Note Histoire'].isnull()), 'Note Histoire']=\
    dred.ix[testRegFrance,[MHT1]+[MHT2]].mean(axis=1)

# If the previous step was not enough, go and fetch the values in marks from 'Première':
# Compute 'Moyenne Histoire/Géographie Tale' when possible
MHGT1='Moyenne candidat en Histoire/Géographie Trimestre 1'
MHGT2='Moyenne candidat en Histoire/Géographie Trimestre 2'

dred.loc[testRegFrance & (dred['Note Histoire'].isnull()), 'Note Histoire']=\
    dred.ix[testRegFrance,[MHGT1]+[MHGT2]].mean(axis=1)
    
## =======================================
# Column 'Note Bac Francais''
# =======================================

# Create new column 'Note Bac Francais' in dred
dred.loc[:,'Note Bac Francais']=np.nan

# Fill in for the case of
NOF="Note à l'épreuve de Oral de Français (épreuve anticipée)"
NEF="Note à l'épreuve de Ecrit de Français (épreuve anticipée)"
NOFL="Note à l'épreuve de Oral Français et littérature (épreuve anticipée)"
NEFL="Note à l'épreuve de Ecrit Français et littérature (épreuve anticipée)"

dred.loc[testRegFrance,'Note Bac Francais']=\
    dred.ix[testRegFrance,[NOF,NEF, NOFL, NEFL]].mean(axis=1)

# fill in of 'Note Bac Francais' with 10 for those whose marks we do not have
dred.loc[testRegFrance & (dred['Note Bac Francais'].isnull()) ,'Note Bac Francais']=10
# # [2017] NB: 266 cases with null 'Note Bac Francais'
# dred[dred['Note Bac Francais'].isnull()].shape

## ===================================================
# Computation of 'ClassA' and 'ClassH'
# ===================================================

# Introduce a 'ClassA' column indicating the relative ranking in the class (Anglais)
dred.loc[:,'ClassA']=np.nan

# Introduce a 'ClassH' column indicating the relative ranking in the class (Histoire/Géographie)
dred.loc[:,'ClassH']=np.nan

# Definition of general case ('LV 1 = Anglais'):
testG=\
    (dred['Langue vivante 1 (note)'].notnull()) & \
    (dred['LV 1 scolarité']=='Anglais') & \
    (dred['Profil du candidat (Code)']==1) & \
    (dred['Classement (Langue vivante 1)'].notnull()) & \
    (dred['Effectif (Langue vivante 1)'].notnull()) & \
    (dred['Classement (Histoire/Géographie)'].notnull()) & \
    (dred['Effectif (Histoire/Géographie)'].notnull())

# [2016] NB: number of elements in this case does not change if we do not test "(dred['Langue vivante 1 (note)'].notnull())"

# Fill in 'ClassA' for 'testG'
dred.loc[testG,'ClassA']=\
    (dred.ix[testG,'Classement (Langue vivante 1)'])/(dred.ix[testG,'Effectif (Langue vivante 1)'])

# Fill in 'ClassH' for 'testG'
dred.loc[testG,'ClassH']=\
    (dred.ix[testG,'Classement (Histoire/Géographie)'])/(dred.ix[testG,'Effectif (Histoire/Géographie)'])

# Definition of case 'LV 2 = Anglais':
testH=\
    (dred['LV 2 scolarité']=='Anglais') & \
    (dred['Profil du candidat (Code)']==1) & \
    (dred['Classement (Langue vivante 2)'].notnull()) & \
    (dred['Effectif (Langue vivante 2)'].notnull()) & \
    (dred['Classement (Histoire/Géographie)'].notnull()) & \
    (dred['Effectif (Histoire/Géographie)'].notnull())

# Fill in 'ClassA' for 'testH'
dred.loc[testH,'ClassA']=\
    (dred.ix[testH,'Classement (Langue vivante 2)'])/(dred.ix[testH,'Effectif (Langue vivante 2)'])

# Fill in 'ClassH' for 'testH'
dred.loc[testH,'ClassH']=\
    (dred.ix[testH,'Classement (Histoire/Géographie)'])/(dred.ix[testH,'Effectif (Histoire/Géographie)'])

# Introduce testAdd ( ("En Terminal" | "Scolarisé dans le supérieur en France" | "Non scolarisé") & ("Note Anglais" is available) & ("Note Histoire" is available) )
testAdd=\
    testRegFrance &\
    (dred['Note Anglais'].notnull()) &\
    (dred['Note Histoire'].notnull()) &\
    (dred['ClassA'].isnull()) &\
    (dred['ClassH'].isnull())


# For testAdd, get a proxy of 'ClassA' and 'ClassH' from the ratio between 'Note Anglais' and 'Note Histoire' and 20.

# Fill in 'ClassA' and 'ClassH' for 'testAdd'
dred.loc[testAdd,'ClassA']=1-(dred.ix[testAdd,'Note Anglais'])/20
dred.loc[testAdd,'ClassH']=1-(dred.ix[testAdd,'Note Histoire'])/20

## ===================================================
# Computation of the overall mark for 'cas G'
# ===================================================

# Create new column 'Note G' in dred
dred.loc[:,'Note G']=np.nan

# # "Union" of testG, testH and testAdd
testReg=\
    (dred['ClassA'].notnull()) &\
    (dred['ClassH'].notnull()) &\
    (dred['Note Bac Francais'].notnull()) &\
    (dred['Bonus'].notnull())


# Compute the overall mark for 'testCond'
def ComputeNoteGv1(testCond,dred):
    dred.loc[testCond,'Note G']=\
    5*(1+(dred.ix[testCond,'ClassH']<=0.25)+(dred.ix[testCond,'ClassH']<=0.5)+(dred.ix[testCond,'ClassH']<=0.75))+\
    5*(1+(dred.ix[testCond,'ClassA']<=0.25)+(dred.ix[testCond,'ClassA']<=0.5)+(dred.ix[testCond,'ClassA']<=0.75))+\
    5*(dred.ix[testCond,'Avis du CE (Code)']-2)+\
    dred.ix[testCond,'Niveau de la classe'].map( {'Faible': -1, 'Assez bon': 0, 'Moyen': 1, 'Bon':2, 'Très bon':3})+\
    1+(dred.ix[testCond,'Note Bac Francais']>=5)+(dred.ix[testCond,'Note Bac Francais']>=10)+(dred.ix[testCond,'Note Bac Francais']>=15)+\
    dred.ix[testCond,'Option internationale (Code)'].map( {'0': 0,'B':4,'1': 4, '2': 4})
# [2016] NB : 'Avis du CE (Code)' - un seul cas avec avis 'défavorable'

def ComputeNoteGv2(testCond,dred):
    dred.loc[testCond,'Note G']=\
    20*(1-dred.ix[testCond,'ClassH'])+\
    20*(1-dred.ix[testCond,'ClassA'])+\
    dred.ix[testCond & (dred['Note Bac Francais'].notnull()),'Note Bac Francais']/5+\
    dred.ix[testCond,'Bonus']
# [2017] avec cette normalisation, 'Note Bac Francais' ajoute entre 0 et 4 points !

ComputeNoteGv2(testReg, dred)

# 'Note G' is computed from 
# 1/ 'ClassH' (up to 20 marks)
# 2/ 'ClassA' (up to 20 marks)
# 3/ 'Note Bac Francais' (1 to 4 marks)
# 4/ 'Bonus' (up to 17) taking into account (for 'testTerminale'):
# _ 'Avis du CE (Code)' (-5 to 10 marks)
# _ 'Niveau de la classe' (-1 to 3 marks)
# _ 'Option européenne (Code)' (0 to 4 marks)
#
# so possible max = 61.

# Find minimum value of 'Note G'
# min(dred.ix[testReg,'Note G'])

# Find maximum value of 'Note G'
# max(dred.ix[testReg,'Note G'])

# Display histogram
# dred.ix[testReg,'Note G'].hist(bins=20, range=(10,60))
#P.show()

# Number of columns: 
# dred.ix[dred['Note G'].notnull(),:].shape

## =========================================================
# Statistics and graphs on results
# =========================================================

# # What remains:
# testNoteG=dred['Note G'].isnull()
# 
# # Find minimum value of 'Note G'
# min(dred['Note G'])
# 
# # Find maximum value of 'Note G'
# max(dred['Note G'])
# 
# # Display histogram - 'Note G'
# dred['Note G'].hist(bins=40, range=(5,56))
# P.show()
# # 
# # # Display histogram - 'Note Anglais'
# # dred['Note Anglais'].hist(bins=20, range=(0,20))
# # P.show()
# 
# # Display histogram - 'Note Histoire'
# dred['Note Histoire'].hist(bins=20, range=(0,20))
# P.show()

# #  Display histogram - 'Distribution 'Note G' Garcons/filles'
# dG = dred[ dred['Sexe']=='M']
# dF = dred[ dred['Sexe']=='F']
# 
# dF['Note G'].hist(bins=40, range=(5,56))
# dG['Note G'].hist(bins=40, range=(5,56))
# P.show()

# # Weird things:
# testW = ((dred['Note Anglais'].isnull())&( ( (dred['LV 1 scolarité']=='Anglais') & (dred['Moyenne LV 1 Tale'].notnull()) ) | ( (dred['LV 2 scolarité']=='Anglais') & (dred['Moyenne LV 2 Tale'].notnull()) )))
# # No such elements!

## =========================================================
# Export 
# =========================================================
# Final result: 
# _ a list of the applicants...
# _ ... sorted according to 'Note G' ...
# _ ... with only a selection of relevant columns!
# 

# Sorting the elements according to 'Note G':
dsort=dred.sort_values('Note G', ascending=False)

# # Retrouver les colonnes qui commencent par 'Classement'
# [ w for w in dred.columns if w.startswith('Classement')]

# # Retrouver les colonnes qui comportent 'établissement'
# [ w for w in dred.columns if "établissement" in w]

# Columns that we want to keep:
listExport=[\
'Note G',\
'Numéro',\
'Nom',\
'Prénom',\
'Date de naissance',\
'Profil du candidat',\
'Libellé établissement',\
'Département établissement',\
'Type de formation',\
'Série/Domaine/Filière',\
'Option internationale',\
'Niveau de la classe',\
'Avis du CE',\
'Note Anglais',\
'Note Histoire'
]

testNoteG=dred['Note G'].notnull()
# Dataframe of those we want to export
dsortBis=dsort.ix[testNoteG,listExport]

# Export to FOUTPUT
dsortBis.to_csv(FOUTPUT,sep=';',decimal=',', encoding='utf-8')

testNoNoteG=dred['Note G'].isnull()
# Those cases that remain:
drest=dsort.ix[testNoNoteG,:]
drest.to_csv(FSPECIAL,sep=';',decimal=',', encoding='utf-8')

## =====================================================
# Appendices
# =====================================================

# What remains:
# testNoteG=(dred['Note G'].isnull())

# Find list columns we want to keep:
# listExport =\
# list(dred.columns[2:10])+[dred.columns[11]]+list(dred.columns[17:26])+list(dred.columns[32:44])+list(dred.columns[70:79])

# Entries that we want to keep:
# listExport=[
# 'Numéro',
# 'nom',
# 'Prénom',
# 'Date de naissance',
# 'Profil du candidat',
# 'Profil du candidat (Code)',
# 'Type établissement',
# 'Type établissement (Code)',
# 'Libellé établissement'
# ]
#

#dExport1 = dred.ix[testNoteG,listExport]

#dExport1.to_csv('C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Bi-licence_Franzi/Data_bi-licence/ExportNoNoteGv2.csv',sep=';')

#dExport.to_csv('C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Bi-licence_Franzi/Data_bi-licence/ExportData.csv',sep=';')

# dExportCompliques=dExport[(dExport['Profil du candidat (Code)']!=1)&(dExport['Profil du candidat (Code)']!=11)]
# 
# dExportCompliques.to_csv('C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Bi-licence_Franzi/Data_bi-licence/ExportCompliques.csv',sep=';')
# 
# dExportTale=dExport[dExport['Profil du candidat (Code)']==1]
# 
# dExportTale.to_csv('C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Bi-licence_Franzi/Data_bi-licence/ExportTale.csv',sep=';')