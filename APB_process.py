import pandas as pd
import numpy as np
import pylab as P

FINPUT='C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Bi-licence_Franzi/Data_bi-licence/datav2-completing-cor2.csv'

FOUTPUT='C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Bi-licence_Franzi/Data_bi-licence/ExportSorted.csv'

# For .read_csv, always use header=0 when you know row 0 is the header row
#df = pd.read_csv('C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Bi-licence_Franzi/Data_bi-licence/datav2.csv',sep=';',decimal=',',header=0)
df = pd.read_csv(FINPUT,sep=';',decimal=',',header=0)
# sep=';' to use only ";" as delimiters
# decimal=',' to use "," as decimal point (and not "." (default behavior))

# Keep only those 
# _ whose application is confirmed ('Candidature validée (Code)'==1)
# _ whose series are not 'STI', 'STG' or 'P'
dred=df[(df['Candidature validée (Code)']==1) & (df['Série (Code)']!='STI') &  (df['Série (Code)']!='STG') & (df['Série (Code)']!='P')]
# NB: it would also be possible to sort according to 'Type de classe (code)' (and keep only those with type != 2).


# =======================================
#
# Introduce testRegFrance ("En Terminal" ou "Scolarisé dans le supérieur en France" ou "Non scolarisé")
testRegFrance=\
(dred['Profil du candidat (Code)']==1) | \
(dred['Profil du candidat (Code)']==11) | \
(dred['Profil du candidat (Code)']==15)

# Estimation of the mean 'Avis du CE (Code)':
# dred['Avis du CE (Code)'].mean()

# Estimation of the mean 'Niveau de la classe':
# dred['Niveau de la classe'].map( {'Faible': -1, 'Assez bon': 0, 'Moyen': 1, 'Bon':2, 'Très bon':3}).mean()

# If 'Avis du CE (Code)' is not available, fill in with 'Favorable'
dred.ix[testRegFrance,'Avis du CE (Code)']=dred.ix[testRegFrance,'Avis du CE (Code)'].fillna(3)

# If 'Niveau de la classe' is not available, fill in with 'Moyen'
dred.ix[testRegFrance,'Niveau de la classe']=dred.ix[testRegFrance,'Niveau de la classe'].fillna('Moyen')

# If 'Option européenne (Code)' is not available, fill in with 'A'
dred.ix[testRegFrance,'Option européenne (Code)']=dred.ix[testRegFrance,'Option européenne (Code)'].fillna('A')



# =======================================
# 
# Create new column 'Note G' in dred
dred.loc[:,'Note G']=np.nan

# =======================================
# Completion of column 'Note Anglais'
# =======================================

# Create new column 'Note Anglais' in dred
dred.loc[:,'Note Anglais']=np.nan

# Create new column 'Moyenne LV 1 Tale' in dred
dred.loc[:,'Moyenne LV 1 Tale']=np.nan

# Fill in column 'Moyenne LV 1 Tale' in dred
dred.ix[testRegFrance,'Moyenne LV 1 Tale']=\
dred.ix[testRegFrance,[dred.columns[79]]+[dred.columns[123]]].mean(axis=1)
# NB: dred.columns[79] yields 'Moyenne candidat en Langue vivante 1 Trimestre 1' (colonne HY)
# NB: dred.columns[123] yields 'Moyenne candidat en Langue vivante 1 Trimestre 2'

# Create new column 'Moyenne LV 2 Tale' in dred
dred.loc[:,'Moyenne LV 2 Tale']=np.nan

# Fill in column 'Moyenne LV 2 Tale' in dred
dred.ix[testRegFrance,'Moyenne LV 2 Tale']=\
dred.ix[testRegFrance,[dred.columns[83]]+[dred.columns[127]]].mean(axis=1)
# NB: dred.columns[83] yields 'Moyenne candidat en Langue vivante 1 Trimestre 1' (colonne HY)
# NB: dred.columns[127] yields 'Moyenne candidat en Langue vivante 1 Trimestre 2'

# First run for 'Note Anglais': if 'LV 2 (Code)'==2 (i.e. LV 2 'Anglais'), then set 'Note Anglais' = 'Langue vivante 2 (note)'
dred.ix[testRegFrance & (dred['LV 2 (Code)']==2),'Note Anglais']=dred.ix[testRegFrance & (dred['LV 2 (Code)']==2),'Langue vivante 2 (note)']

# Second run for 'Note Anglais': if 'LV 1 scolarité'=='Anglais', then set 'Note Anglais' = 'Langue vivante 1 (note)'
dred.ix[testRegFrance & (dred['LV 1 scolarité']=='Anglais'),'Note Anglais']=\
dred.ix[testRegFrance & (dred['LV 1 scolarité']=='Anglais'),'Langue vivante 1 (note)']

# NB: on the available data, there is no case of ( ('LV 2 (Code)'==2) & ('LV 1 (Code)'==2) ), but running 'LV 1 (Code)'==2 in second ensure that these marks would take precedence. 

# Third run: fill in those that remain, using the marks from Tale:
# if (dred['LV 1 scolarité']=='Anglais'), then update 'Note Anglais' with 'Moyenne LV 1 Tale'.
dred.ix[testRegFrance & (dred['Note Anglais'].isnull()) & (dred['LV 1 scolarité']=='Anglais'),'Note Anglais']=\
dred.ix[testRegFrance & (dred['Note Anglais'].isnull()) & (dred['LV 1 scolarité']=='Anglais'),'Moyenne LV 1 Tale']

# Fourth run: fill in those that remain, using the marks from Tale:
# if (dred['LV 2 scolarité']=='Anglais'), then update 'Note Anglais' with 'Moyenne LV 2 Tale'.
dred.ix[testRegFrance & (dred['Note Anglais'].isnull()) & (dred['LV 2 scolarité']=='Anglais'),'Note Anglais']=\
dred.ix[testRegFrance & (dred['Note Anglais'].isnull()) & (dred['LV 2 scolarité']=='Anglais'),'Moyenne LV 2 Tale']

# =======================================
# Completion of column 'Histoire/Géographie (note)'
# =======================================

# Create new column 'Moyenne Histoire/Géographie Tale' in dred
dred.loc[:,'Moyenne Histoire/Géographie Tale']=np.nan

# Compute 'Moyenne Histoire/Géographie Tale' when possible
dred.ix[testRegFrance,'Moyenne Histoire/Géographie Tale']=\
dred.ix[testRegFrance,[dred.columns[87]]+[dred.columns[131]]].mean(axis=1)
# NB: dred.columns[87] yields 'Moyenne candidat en Histoire/Géographie Trimestre 1' (colonne HU)
# NB: dred.columns[131] yields 'Moyenne candidat en Histoire/Géographie Trimestre 2'

# If 'Histoire/Géographie (note)' is not available, fill in with 'Moyenne Histoire/Géographie Tale'
dred.ix[testRegFrance,'Histoire/Géographie (note)']=dred.ix[testRegFrance,'Histoire/Géographie (note)'].fillna(dred['Moyenne Histoire/Géographie Tale'])

# If the previous step was not enough, go and fetch the values in marks from 'Première':
# Compute 'Moyenne Histoire/Géographie Tale' when possible
dred.ix[testRegFrance & (dred['Histoire/Géographie (note)'].isnull()),'Moyenne Histoire/Géographie Tale']=\
dred.ix[testRegFrance & (dred['Histoire/Géographie (note)'].isnull()),[dred.columns[228]]+[dred.columns[272]]].mean(axis=1)
# NB: dred.columns[228] is 'Moyenne candidat en Histoire/Géographie Trimestre 1'
# NB: dred.columns[272] is 'Moyenne candidat en Histoire/Géographie Trimestre 2'

dred.ix[testRegFrance & (dred['Histoire/Géographie (note)'].isnull()),'Histoire/Géographie (note)']=\
dred.ix[testRegFrance & (dred['Histoire/Géographie (note)'].isnull()),'Moyenne Histoire/Géographie Tale']

# =======================================
# Column 'Note Bac Francais''
# =======================================

# Create new column 'Note Bac Francais' in dred
dred.loc[:,'Note Bac Francais']=np.nan

# Fill in for the case of
dred.ix[testRegFrance,'Note Bac Francais']=dred.ix[testRegFrance,list(dred.columns[59:61])].mean(axis=1)
# NB: dred.columns[59] is 'Note à l'épreuve de Oral de Français (épreuve anticipée)'
# NB: dred.columns[60] is 'Note à l'épreuve de Ecrit de Français (épreuve anticipée)'

# ===================================================
# Computation of the overall mark for 'cas G'
# ===================================================
#

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

# NB: number of elements in this case does not change if we do not test "(dred['Langue vivante 1 (note)'].notnull())"

# Fill in 'ClassA' for 'testG'
dred.ix[testG,'ClassA']=(dred.ix[testG,'Classement (Langue vivante 1)'])/(dred.ix[testG,'Effectif (Langue vivante 1)'])

# Fill in 'ClassH' for 'testG'
dred.ix[testG,'ClassH']=(dred.ix[testG,'Classement (Histoire/Géographie)'])/(dred.ix[testG,'Effectif (Histoire/Géographie)'])

# Definition of case 'LV 2 = Anglais':
testH=\
(dred['LV 2 scolarité']=='Anglais') & \
(dred['Profil du candidat (Code)']==1) & \
(dred['Classement (Langue vivante 2)'].notnull()) & \
(dred['Effectif (Langue vivante 2)'].notnull()) & \
(dred['Classement (Histoire/Géographie)'].notnull()) & \
(dred['Effectif (Histoire/Géographie)'].notnull())

# Fill in 'ClassA' for 'testH'
dred.ix[testH,'ClassA']=(dred.ix[testH,'Classement (Langue vivante 2)'])/(dred.ix[testH,'Effectif (Langue vivante 2)'])

# Fill in 'ClassH' for 'testH'
dred.ix[testH,'ClassH']=(dred.ix[testH,'Classement (Histoire/Géographie)'])/(dred.ix[testH,'Effectif (Histoire/Géographie)'])

# "Union" of testG and testH
testReg=(testG | testH)

# Compute the overall mark for the 'general case'
dred.ix[testReg,'Note G']=\
5*(1+(dred.ix[testReg,'ClassH']<=0.25)+(dred.ix[testReg,'ClassH']<=0.5)+(dred.ix[testReg,'ClassH']<=0.75))+\
5*(1+(dred.ix[testReg,'ClassA']<=0.25)+(dred.ix[testReg,'ClassA']<=0.5)+(dred.ix[testReg,'ClassA']<=0.75))+\
5*(dred.ix[testReg,'Avis du CE (Code)']-2)+\
dred.ix[testReg,'Niveau de la classe'].map( {'Faible': -1, 'Assez bon': 0, 'Moyen': 1, 'Bon':2, 'Très bon':3})+\
1+(dred.ix[testReg,'Note Bac Francais']>=5)+(dred.ix[testReg,'Note Bac Francais']>=10)+(dred.ix[testReg,'Note Bac Francais']>=15)+\
dred.ix[testReg,'Option européenne (Code)'].map( {'A': 0,'B':4,'E': 4, 'I': 4})
# NB : l. 179 - un seul cas avec avis 'défavorable'
#
# 'Note G' is computed from 
# 1/ ranking in 'Histoire/Géographie' (5 to 20 marks)
# 2/ ranking in 'Anglais' (5 to 20 marks)
# 3/ 'Avis du CE (Code)' (-5 to 10 marks)
# 4/ 'Niveau de la classe' (-1 to 3 marks)
# 5/ 'Note Bac Francais' (1 to 4 marks)
# 6/ 'Option européenne (Code)' (0 to 4 marks)
# so possible min = 5 and possible max = 61.

# Find minimum value of 'Note G'
# min(dred.ix[testReg,'Note G'])

# Find maximum value of 'Note G'
# max(dred.ix[testReg,'Note G'])

# Display histogram
# dred.ix[testReg,'Note G'].hist(bins=20, range=(10,60))
#P.show()

# Number of columns: 
# dred.ix[dred['Note G'].notnull(),:].shape

# =======================================
# Computation of bounds for ranking:
# =======================================
# General idea: once the "testReg" case is treated, 
# try to evaluate the score for those whose ranking is not available, 
# using bounds defined from the "testReg" case.
#
# Computation of means 'ClassA'
m1A=dred.ix[dred['ClassA']<=0.25,'Note Anglais'].mean()

m2A=dred.ix[(dred['ClassA']>0.25) & (dred['ClassA']<=0.5),'Note Anglais'].mean()

m3A=dred.ix[(dred['ClassA']>0.5) & (dred['ClassA']<=0.75),'Note Anglais'].mean()

m4A=dred.ix[(dred['ClassA']>0.75),'Note Anglais'].mean()

# Computation of bounds 'ClassA'
b1A=(m1A+m2A)/2
b2A=(m2A+m3A)/2
b3A=(m3A+m4A)/2

# Computation of means 'ClassH'
m1H=dred.ix[dred['ClassH']<=0.25,'Histoire/Géographie (note)'].mean()

m2H=dred.ix[(dred['ClassH']>0.25) & (dred['ClassH']<=0.5),'Histoire/Géographie (note)'].mean()

m3H=dred.ix[(dred['ClassH']>0.5) & (dred['ClassH']<=0.75),'Histoire/Géographie (note)'].mean()

m4H=dred.ix[(dred['ClassH']>0.75),'Histoire/Géographie (note)'].mean()

# Computation of bounds 'ClassH'
b1H=(m1H+m2H)/2
b2H=(m2H+m3H)/2
b3H=(m3H+m4H)/2

# Introduce testAdd ( ("En Terminal" | "Scolarisé dans le supérieur en France") & ("Note G" is missing) & ("Note Anglais" is available) & ("Histoire/Géographie (note)" is available) )
testAdd=\
testRegFrance &\
(dred['Note G'].isnull()) &\
(dred['Note Anglais'].notnull()) &\
(dred['Histoire/Géographie (note)'].notnull()) &\
(dred['Note Bac Francais'].notnull())

# Compute the overall mark for the "testAdd" case
dred.ix[testAdd,'Note G']=\
5*(1+(dred.ix[testAdd,'Note Anglais']>=b3A)+(dred.ix[testAdd,'Note Anglais']>=b2A)+(dred.ix[testAdd,'Note Anglais']>=b1A))+\
5*(1+(dred.ix[testAdd,'Histoire/Géographie (note)']>=b3H)+(dred.ix[testAdd,'Histoire/Géographie (note)']>=b2H)+(dred.ix[testAdd,'Histoire/Géographie (note)']>=b1H))+\
5*(dred.ix[testAdd,'Avis du CE (Code)']-2)+\
dred.ix[testAdd,'Niveau de la classe'].map( {'Faible': -1, 'Assez bon': 0, 'Moyen': 1, 'Bon':2, 'Très bon':3})+\
1+(dred.ix[testAdd,'Note Bac Francais']>=5)+(dred.ix[testAdd,'Note Bac Francais']>=10)+(dred.ix[testAdd,'Note Bac Francais']>=15)+\
4*dred.ix[testAdd,'Option européenne (Code)'].map( {'A': 0,'B':1,'E': 1, 'I': 1})
# For 'Note Bac Francais': 
# _ less than 5: 1 mark
# _ between 5 and 10: 2 mark
# _ between 10 and 15: 3 mark
# _ more than 15: 4 mark

# =========================================================
# Statistics and graphs on results
# =========================================================

# # What remains:
testNoteG=dred['Note G'].isnull()
# 
# # Find minimum value of 'Note G'
# min(dred['Note G'])
# 
# # Find maximum value of 'Note G'
# max(dred['Note G'])
# 
# # Display histogram - 'Note G'
# dred['Note G'].hist(bins=30, range=(5,65))
# P.show()

# # Display histogram - 'Note Anglais'
# dred['Note Anglais'].hist(bins=20, range=(0,20))
# P.show()

# Display histogram - 'Histoire/Géographie (note)'
# dred['Histoire/Géographie (note)'].hist(bins=20, range=(0,20))
# P.show()

# # Weird things:
# testW = ((dred['Note Anglais'].isnull())&( ( (dred['LV 1 scolarité']=='Anglais') & (dred['Moyenne LV 1 Tale'].notnull()) ) | ( (dred['LV 2 scolarité']=='Anglais') & (dred['Moyenne LV 2 Tale'].notnull()) )))
# # No such elements!

# =========================================================
# Export 
# =========================================================
# Final result: 
# _ a list of the applicants...
# _ ... sorted according to 'Note G' ...
# _ ... with only a selection of relevant columns!
# 

# Sorting the elements according to 'Note G':
dsort=dred.sort_values('Note G', ascending=False)

# Columns that we want to keep:
listExport=[\
'Note G',\
'Numéro',\
'nom',\
'Prénom',\
'Date de naissance',\
'Profil du candidat',\
'Type établissement',\
'Libellé établissement',\
'Type de formation',\
'Série/Domaine/Filière',\
'Option européenne',\
'Niveau de la classe',\
'Avis du CE',\
'Langue vivante 1 (note)',\
'Histoire/Géographie (note)',\
dred.columns[59],dred.columns[60]\
]
# dred.columns[59],dred.columns[60] are 'Note à l'épreuve de Oral de Français (épreuve anticipée)' and 'Note à l'épreuve de Ecrit de Français (épreuve anticipée)', respectively.

# Dataframe of those we want to export
dsortBis=dsort.ix[:,listExport]

# # Export to 'ExportSorted.csv'
# dsortBis.to_csv('C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Bi-licence_Franzi/Data_bi-licence/ExportSorted.csv',sep=';')

# =====================================================
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