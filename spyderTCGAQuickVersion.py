
# coding: utf-8

# # DeepLearning and MKL for TCGA Data
# %%

## Librerias
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
# %%
# ## Origen de los datos
# 
# Los datos provienen de __[FireHose Brad GDAC](https://gdac.broadinstitute.org/)__ y del paquete de R __[TCGA-Assembler](https://github.com/compgenome365/TCGA-Assembler-2)__. 
# 
# En el archivo de R *Depuracion_TCGA-Assembler_Integrator.R* se realizo la consolidacion de los archivos. Se trabajo **cnv**, **meth** y **mrna**. Por tal motivo los archivos quedaron listo para usar, sin embargo, se requiere de preprocesameinto para **mrna**. Los genes de este dataSet tienen una varianza diversa, es decir, varianzas cercanas a cero y otras que superan el millon.
# 
# El siguiente documento vendra organizado por: <br>
# 1. [Funciones](#Funciones)
# 2. [PreprossData](#PreprossData)
# 3. [ANN](#ANN)

# %%
os.chdir('C:/Users/da.salazarb/Google Drive/Tutorial_2018-10/03_Resultados/DataTCGA')
## Cual es la ruta actual?
os.getcwd()
# %%
# ******************************************************************************************************************************************

# ### Cargar archivos

# Pasar a 
# 1. [Funciones](#Funciones)
# 2. [PreprossData](#PreprossData)
# 3. [ANN](#ANN)

# ******************************************************************************************************************************************

# ## Funciones 
# [Retornar al inicio](#Data)

# In[28]:


#Normalizar datos
def normalizar(tcga):
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler(with_mean=True, with_std=True)
    sc_x.fit(tcga)
    x_train_s = sc_x.transform(tcga)
    x_train_s=pd.DataFrame(x_train_s, index=tcga.index, columns=tcga.columns)
    return x_train_s

#Normalizar en rango [0,1]
def scalarRango(tcga):
    from sklearn.preprocessing import MinMaxScaler
    sc_x = MinMaxScaler()
    sc_x.fit(tcga)
    x_train_s = sc_x.transform(tcga)
    x_train_s=pd.DataFrame(x_train_s, index=tcga.index, columns=tcga.columns)
    return x_train_s

# Seleccion de variables para varianzas pequeñas
def varianceSelection(X, THRESHOLD = 10):
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=THRESHOLD)
    sel.fit_transform(X)
    return X[[c for (s, c) in zip(sel.get_support(), X.columns.values) if s]]

#Para cargar nuevamente los archivos de forma rapida
def reRunFiles(norm=True, varX=False, logSurv=False, varianza=False, randVar=(False, 0), pcaVar=(False, 0), preVitaVar=False, setTrainTest=False, preMRmrVar=False):
    '''
    varX: True para eliminar la primera variable que puede ser el identificador del paciente
    logSurv: True para sacar logaritmo de 'y' o dsurv
    varianza: True para eliminar variables por varianza segun threshold
    randVar: True para elegir variables, int para escoger una cantidad de variables
    PreVar: True para elegir 100 variables aleatorias predefinidas con poca correlacion entre ellas
    setTrainTest:  True para retornar train y dev set
    
    '''
    mrna=pd.read_table("TCGA-Integrator/mrnaPrePros.txt")
    cnv=pd.read_table("TCGA-Integrator/cnvPrePros.txt")
    meth=pd.read_table("TCGA-Integrator/methPrePros.txt")
    dsurv=pd.read_table("TCGA-Integrator/dsurvPrePros.txt")
    protein=pd.read_table("TCGA-Integrator/dataTCGA/originalData/new_gbm_Protein.txt", sep=",", index_col=0)
    
    # Se elimina la var X
    if varX == True:
        mrna=mrna.iloc[:,1:]
        cnv=cnv.iloc[:,1:]
        meth=meth.iloc[:,1:]
        dsurv=dsurv.iloc[:,1:]
    
    if norm == True:
        # Normalizar
        mrna=normalizar(mrna)
        cnv=normalizar(cnv)
        meth=normalizar(meth)
        protein=normalizar(protein)
    else:
        # NormalizarRango
        mrna=scalarRango(mrna)
        cnv=scalarRango(cnv)
        meth=scalarRango(meth)
        protein=scalarRango(protein)
    
    # Los archivos vienen con una columna "X" que contiene los indices de los pacientes
    # si varX == True -> se elimina esa primera columna
    
#    if preVar == True:
#        #Lista de 100 variables aleatoria
#        mrnaLista=['JPH1', 'MT1X', 'NR2E3', 'PEX5L', 'CHST6', 'VEGFC', 'TPSD1', 'TMUB1', 'ACER1', 'OXER1', 'VAV3', 'HAND2', 'CYP11A1', 'HS3ST3B1', 'HIF3A', 'RPL21', 'LDLR', 'MUC15', 'PON2', 'WDR72', 'SH2D4B', 'SLC7A2', 'S100A3', 'RCVRN', 'USP29', 'AQP2', 'FAM83H', 'SPAG4', 'TP53I11', 'SORCS3', 'GLT1D1', 'CCIN', 'TFAP2B', 'GAD1', 'EPHX3', 'KRT17', 'GUCY2D', 'DIRAS2', 'HIST1H2AB']
#        mrna=mrna[mrnaLista]
#        
#        #Lista de 100 variables aleatoria
#        cnvLista=['FAM8A1_CHR6_pos_cnv', 'LSP1P3_CHR5_pos_cnv', 'CDCP1_CHR3_neg_cnv', 'RPS14_CHR5_neg_cnv', 'THOC2_CHRX_neg_cnv', 'GPR6_CHR6_pos_cnv', 'LRRC10B_CHR11_pos_cnv', 'TSSK3_CHR1_pos_cnv', 'ATP6V1A_CHR3_pos_cnv', 'PLGLB2_CHR2_pos_cnv', 'SNORD68_CHR16_pos_cnv', 'TRIM66_CHR11_neg_cnv', 'UBE2N_CHR12_neg_cnv', 'TAOK1_CHR17_pos_cnv', 'ARHGAP10_CHR4_pos_cnv', 'CCDC9_CHR19_pos_cnv', 'IFNA13_CHR9_neg_cnv', 'DOK7_CHR4_pos_cnv', 'RBM44_CHR2_pos_cnv', 'SNRNP48_CHR6_pos_cnv', 'B3GNT3_CHR19_pos_cnv', 'ALOX5AP_CHR13_pos_cnv', 'ATRX_CHRX_neg_cnv', 'LOC643650_CHR10_neg_cnv', 'SREBF2_CHR22_pos_cnv', 'SNAI1_CHR20_pos_cnv', 'SHC3_CHR9_neg_cnv', 'CBX3P2_CHR18_neg_cnv', 'GOLGA6C_CHR15_pos_cnv', 'MAGEB1_CHRX_pos_cnv', 'ARHGAP42_CHR11_pos_cnv', 'KCNK9_CHR8_neg_cnv', 'SOS1_CHR2_neg_cnv', 'SERPINA10_CHR14_neg_cnv', 'CLEC18A_CHR16_pos_cnv', 'PDCD1LG2_CHR9_pos_cnv', 'ROR1_CHR1_pos_cnv', 'APOBEC1_CHR12_neg_cnv', 'PRKD1_CHR14_neg_cnv', 'RNF41_CHR12_neg_cnv', 'COX6A2_CHR16_neg_cnv', 'RTL8B_CHRX_neg_cnv', 'MINDY1_CHR1_neg_cnv', 'F2RL2_CHR5_neg_cnv', 'LOXL2_CHR8_neg_cnv', 'LOC100498859_CHR3_pos_cnv', 'GK2_CHR4_neg_cnv', 'SNORD1A_CHR17_pos_cnv', 'LGSN_CHR6_neg_cnv', 'POLI_CHR18_pos_cnv', 'CD24_CHRY_neg_cnv', 'KLHL11_CHR17_neg_cnv']
#        cnv=cnv[cnvLista]
#        
#        #Lista de 100 variables aleatoria
#        methLista=['IL6_Met', 'KCNF1_Met', 'OSTBETA_Met', 'ITGA8_Met', 'ALOX15_Met', 'CCNL1_Met', 'CCM2_Met', 'CXCL1_Met', 'KCNH8_Met', 'RASD2_Met', 'TMEM49_Met', 'HLA.A_Met']
#        meth=meth[methLista]
#        
#        #Lista de 100 variables aleatoria
#        proteinLista=['XRCC5_Protein', 'FASN_Protein', 'CDKN1A_Protein', 'SMAD1_Protein', 'TFRC_Protein', 'BCL2_Protein', 'GAPDH_Protein', 'SYK_Protein', 'CDKN1B_Protein', 'NFKB1_Protein', 'PDCD4_Protein', 'RPS6_Protein.2', 'ERBB2_Protein', 'PCNA_Protein', 'MAPK14_Protein.1', 'CDK1_Protein', 'LCK_Protein', 'MYH11_Protein', 'G6PD_Protein', 'ASNS_Protein', 'AXL_Protein', 'AKT1 AKT2 AKT3_Protein.1', 'BECN1_Protein', 'ITGA2_Protein', 'PRKAA1_Protein', 'BRD4_Protein', 'CAV1_Protein', 'IRF1_Protein', 'SETD2_Protein', 'EGFR_Protein.1', 'CDH3_Protein', 'MAP2K1_Protein', 'SRC_Protein', 'MYH9_Protein']
#        protein=protein[proteinLista]
    
    if preVitaVar == True:
        vitaVar=pd.read_table("C:/Users/da.salazarb/Google Drive/Tutorial_2018-10/03_Resultados/DataTCGA/varSelected/vita/vitaVarSelect.txt")
        mrna=mrna[list(vitaVar.iloc[:,0])]
        cnvNames = []
        for i in list(vitaVar.iloc[:,0]):
            [cnvNames.append(s) for s in cnv.columns if i in s]
        cnv=cnv[cnvNames]
        methNames = []
        for i in list(vitaVar.iloc[:,0]):
            [methNames.append(s) for s in meth.columns if i in s]
        meth=meth[methNames]
        
    if preMRmrVar == True:
        MRmrVar=pd.read_table("C:/Users/da.salazarb/Google Drive/Tutorial_2018-10/03_Resultados/DataTCGA/varSelected/mrMR/mrMRoneResultVarSelect.txt")
        mrna=mrna[list(MRmrVar.iloc[:,0])]
        cnvNames = []
        for i in list(MRmrVar.iloc[:,0]):
            [cnvNames.append(s) for s in cnv.columns if i in s]
        cnv=cnv[cnvNames]
        methNames = []
        for i in list(MRmrVar.iloc[:,0]):
            [methNames.append(s) for s in meth.columns if i in s]
        meth=meth[methNames]
    
    # Logaritmo a dsurv
    if logSurv == True:
        dsurv=np.log10(dsurv)
    
    # Correccion de nombres para protein
    nombreProteinIndex=list(protein.index)
    nombreProteinIndex = [w.replace('-', '.') for w in nombreProteinIndex]
    protein.index=nombreProteinIndex
    protein=protein.loc[mrna.index,:]
    
    # Seleccionar por varianza
    if varianza == True:
        mrna=varianceSelection(mrna, THRESHOLD=0.5)
        cnv=varianceSelection(cnv, THRESHOLD=0.01)
        meth=varianceSelection(meth, THRESHOLD=0.001)
        protein=varianceSelection(protein, THRESHOLD=0.1)
    
    # variables aleatorias
    if randVar[0] == True:
        variables=np.random.choice(len(mrna.columns), randVar[1])
        mrna=mrna.iloc[:,variables]
        variables=np.random.choice(len(cnv.columns), randVar[1])
        cnv=cnv.iloc[:,variables]
        variables=np.random.choice(len(meth.columns), randVar[1])
        meth=meth.iloc[:,variables]
        variables=np.random.choice(len(protein.columns), randVar[1])
        protein=protein.iloc[:,variables]
    
    # PCA
    if pcaVar[0] == True:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pcaVar[1])
        mrna = pca.fit_transform(mrna)
        cnv = pca.fit_transform(cnv)
        meth = pca.fit_transform(meth)
        protein = pca.fit_transform(protein)
        
    if setTrainTest == True:
        # Obtener train y dev set de cada profile
        from sklearn.model_selection import train_test_split
        mrna_train, mrna_test, dsurv_train, dsurv_test = train_test_split(pd.DataFrame(mrna), pd.DataFrame(dsurv), test_size=.2, random_state=10)
        cnv_train=pd.DataFrame(cnv).loc[mrna_train.index]
        cnv_test=pd.DataFrame(cnv).loc[mrna_test.index]
        meth_train=pd.DataFrame(meth).loc[mrna_train.index]
        meth_test=pd.DataFrame(meth).loc[mrna_test.index]
        protein_train=pd.DataFrame(protein).loc[mrna_train.index]
        protein_test=pd.DataFrame(protein).loc[mrna_test.index]
        
        mrna_train_index = mrna_train.index
        mrna_train_columns = mrna_train.columns
        mrna_test_index = mrna_test.index
        mrna_test_columns = mrna_test.columns
        cnv_train_index = cnv_train.index
        cnv_train_columns = cnv_train.columns
        cnv_test_index = cnv_test.index
        cnv_test_columns = cnv_test.columns
        meth_train_index = meth_train.index
        meth_train_columns = meth_train.columns
        meth_test_index = meth_test.index
        meth_test_columns = meth_test.columns
        protein_train_index = protein_train.index
        protein_train_columns = protein_train.columns
        protein_test_index = protein_test.index
        protein_test_columns = protein_test.columns
        dsurv_train_index = dsurv_train.index
        dsurv_train_columns = dsurv_train.columns
        dsurv_test_index = dsurv_test.index
        dsurv_test_columns = dsurv_test.columns
        
        print("Dimensiones mrna: {0}".format(mrna.shape))
        print("Dimensiones cnv: {0}".format(cnv.shape))
        print("Dimensiones meth: {0}".format(meth.shape))
        print("Dimensiones dsurv: {0}".format(dsurv.shape))
        print("Dimensiones protein: {0}".format(protein.shape))
        
        # Normalizar
        mrna=np.array(mrna)
        cnv=np.array(cnv)
        meth=np.array(meth)
        protein=np.array(protein)
        dsurv=np.array(dsurv)
        mrna_train=np.array(mrna_train)
        mrna_test=np.array(mrna_test)
        cnv_train=np.array(cnv_train)
        cnv_test=np.array(cnv_test)
        meth_train=np.array(meth_train)
        meth_test=np.array(meth_test)
        protein_train=np.array(protein_train)
        protein_test=np.array(protein_test)
        dsurv_train=np.array(dsurv_train)
        dsurv_test=np.array(dsurv_test)
        
        return mrna, cnv, meth, dsurv, protein, mrna_train, mrna_test, cnv_train, cnv_test, meth_train, meth_test, protein_train, protein_test, dsurv_train, dsurv_test, mrna_train_index, mrna_train_columns, mrna_test_index, mrna_test_columns, cnv_train_index, cnv_train_columns, cnv_test_index, cnv_test_columns, meth_train_index, meth_train_columns, meth_test_index, meth_test_columns, protein_train_index, protein_train_columns, protein_test_index, protein_test_columns, dsurv_train_index, dsurv_train_columns, dsurv_test_index, dsurv_test_columns
    else:
        print("Dimensiones mrna: {0}".format(mrna.shape))
        print("Dimensiones cnv: {0}".format(cnv.shape))
        print("Dimensiones meth: {0}".format(meth.shape))
        print("Dimensiones dsurv: {0}".format(dsurv.shape))
        print("Dimensiones protein: {0}".format(protein.shape))
        
        # Normalizar
        mrna=np.array(mrna)
        cnv=np.array(cnv)
        meth=np.array(meth)
        protein=np.array(protein)
        dsurv=np.array(dsurv)
        
        return mrna, cnv, meth, dsurv, protein

# %% Guardar las pariticiones para iBAG

def guardarParticiones(tcga, fileName, index, columns):
    tcga = pd.DataFrame(tcga, index=index, columns=columns)
    tcga.to_csv(fileName)
    #np.savetxt(fileName, tcga)

# In[29]:


#KFold para obtener los train y test de cada perfil
def kfoldProfiles(tcga, n_splits, seed): 
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    splits = kfold.split(tcga)
    return splits


# ## PreprossData
# [Retornar al inicio](#Data)

# In[30]:

# Cargar datos
mrna, cnv, meth, dsurv, protein = reRunFiles(norm=True, varX=True, logSurv=True, varianza=True, randVar=(False, 0), pcaVar=(False, 0), preVitaVar=False, setTrainTest=False, preMRmrVar=False)

# %%
# Cargar datos con train test_validation set ->> noPCA siScaleRango
mrna, cnv, meth, dsurv, protein, mrna_train, mrna_test, cnv_train, cnv_test, meth_train, meth_test, protein_train, protein_test, dsurv_train, dsurv_test, mrna_train_index, mrna_train_columns, mrna_test_index, mrna_test_columns, cnv_train_index, cnv_train_columns, cnv_test_index, cnv_test_columns, meth_train_index, meth_train_columns, meth_test_index, meth_test_columns, protein_train_index, protein_train_columns, protein_test_index, protein_test_columns, dsurv_train_index, dsurv_train_columns, dsurv_test_index, dsurv_test_columns = reRunFiles(norm=False, varX=True, logSurv=True, varianza=False, randVar=(False, 0), pcaVar=(False, 0), preVitaVar=False, setTrainTest=True, preMRmrVar=False)

# %%
# Cargar datos con train test_validation set ->> siPCA 
mrna, cnv, meth, dsurv, protein, mrna_train, mrna_test, cnv_train, cnv_test, meth_train, meth_test, protein_train, protein_test, dsurv_train, dsurv_test, mrna_train_index, mrna_train_columns, mrna_test_index, mrna_test_columns, cnv_train_index, cnv_train_columns, cnv_test_index, cnv_test_columns, meth_train_index, meth_train_columns, meth_test_index, meth_test_columns, protein_train_index, protein_train_columns, protein_test_index, protein_test_columns, dsurv_train_index, dsurv_train_columns, dsurv_test_index, dsurv_test_columns = reRunFiles(norm=True, varX=True, logSurv=True, varianza=False, randVar=(False, 0), pcaVar=(True, 100), preVitaVar=False, setTrainTest=True, preMRmrVar=False)

# %%

# Cargar datos con train test_validation set ->> noPCA siVitaVar
mrna, cnv, meth, dsurv, protein, mrna_train, mrna_test, cnv_train, cnv_test, meth_train, meth_test, protein_train, protein_test, dsurv_train, dsurv_test, mrna_train_index, mrna_train_columns, mrna_test_index, mrna_test_columns, cnv_train_index, cnv_train_columns, cnv_test_index, cnv_test_columns, meth_train_index, meth_train_columns, meth_test_index, meth_test_columns, protein_train_index, protein_train_columns, protein_test_index, protein_test_columns, dsurv_train_index, dsurv_train_columns, dsurv_test_index, dsurv_test_columns = reRunFiles(norm=True, varX=True, logSurv=True, varianza=False, randVar=(False, 0), pcaVar=(False, 0), preVitaVar=True, setTrainTest=True, preMRmrVar=False)

# %%

# Cargar datos con train test_validation set ->> noPCA noVitaVar simrMRVar
mrna, cnv, meth, dsurv, protein, mrna_train, mrna_test, cnv_train, cnv_test, meth_train, meth_test, protein_train, protein_test, dsurv_train, dsurv_test, mrna_train_index, mrna_train_columns, mrna_test_index, mrna_test_columns, cnv_train_index, cnv_train_columns, cnv_test_index, cnv_test_columns, meth_train_index, meth_train_columns, meth_test_index, meth_test_columns, protein_train_index, protein_train_columns, protein_test_index, protein_test_columns, dsurv_train_index, dsurv_train_columns, dsurv_test_index, dsurv_test_columns = reRunFiles(norm=True, varX=True, logSurv=False, varianza=False, randVar=(False, 0), pcaVar=(False, 0), preVitaVar=False, setTrainTest=True, preMRmrVar=True)

# %%
#path = "C:/Users/da.salazarb/Google Drive/Tutorial_2018-10/03_Resultados/DataTCGA/TCGA-Integrator/traintestTCGA/"
path = "C:/Users/da.salazarb/Google Drive/Tutorial_2018-10/03_Resultados/DataTCGA/SimpleMKLTCGA/dataTCGASimpleMKL/"

# Guardar los archivos
guardarParticiones(tcga = mrna_train, fileName = path + "mrna_train.txt", index = mrna_train_index, columns = mrna_train_columns)
guardarParticiones(tcga = cnv_train, fileName = path + "cnv_train.txt", index = cnv_train_index, columns = cnv_train_columns)
guardarParticiones(tcga = meth_train, fileName = path + "meth_train.txt", index = meth_train_index, columns = meth_train_columns)
guardarParticiones(tcga = dsurv_train, fileName = path + "dsurv_train.txt", index = dsurv_train_index, columns = dsurv_train_columns)
guardarParticiones(tcga = protein_train, fileName = path + "protein_train.txt", index = protein_train_index, columns = protein_train_columns)

guardarParticiones(tcga = mrna_test, fileName = path + "mrna_test.txt", index = mrna_test_index, columns = mrna_test_columns)
guardarParticiones(tcga = cnv_test, fileName = path + "cnv_test.txt", index = cnv_test_index, columns = cnv_test_columns)
guardarParticiones(tcga = meth_test, fileName = path + "meth_test.txt", index = meth_test_index, columns = meth_test_columns)
guardarParticiones(tcga = dsurv_test, fileName = path + "dsurv_test.txt", index = dsurv_test_index, columns = dsurv_test_columns)
guardarParticiones(tcga = protein_test, fileName = path + "protein_test.txt", index = protein_test_index, columns = protein_test_columns)

# %% Aqui se completa el train/test para proteina, ademas se normaliza

protein=pd.read_table("TCGA-Integrator/new_gbm_Protein.txt", sep=",", index_col=0)
nombreProteinIndex=list(protein.index)
nombreProteinIndex = [w.replace('-', '.') for w in nombreProteinIndex]
protein.index=nombreProteinIndex
mrna=pd.read_table("TCGA-Integrator/mrnaPrePros.txt")
mrna=mrna.iloc[:,1:10]
protein=protein.loc[mrna.index,:]
protein=normalizar(protein)

mrna_train=pd.read_table("TCGA-Integrator/traintestTCGA/mrna_train.txt", sep=",", index_col=0)
mrna_train = mrna_train.iloc[:,0:10]

protein_train=protein.loc[mrna_train.index,:]
protein_test=protein.loc[~protein.index.isin(mrna_train.index),:]

guardarParticiones(tcga = protein_train, fileName = "TCGA-Integrator/traintestTCGA/protein_train.txt", index = protein_train.index, columns = protein_train.columns)
guardarParticiones(tcga = protein_test, fileName = "TCGA-Integrator/traintestTCGA/protein_test.txt", index = protein_test.index, columns = protein_test.columns)

# In[48]:
# Correr en python 3.6
# Neural Network
from nnTCGA import multiModelGenLayer, multiModelGenProtLayer, allToDNN

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# %%
## Multi-modal NN mrna, cnv y meth
model = multiModelGenLayer(mrna, cnv, meth)
consolidado = []
for j in range(10,35):
    for i in range(10):
        model.fit([mrna_train, cnv_train, meth_train], dsurv_train,
                  epochs=j, batch_size=10, verbose=0)
        
        y_pred = model.predict([mrna_test, cnv_test, meth_test])
        consolidado.append(mean_squared_error(dsurv_test, y_pred))
    print("epochs: " + str(j) + " Mean-> mseTest: " + str(np.mean(consolidado)) + " Variance: " + str(np.var(consolidado)))
#print("Test MSE:", mean_squared_error(dsurv_test, y_pred))

# %%
## Multi-modal NN mrna, cnv, meth y protein
model = multiModelGenProtLayer(mrna, cnv, meth, protein)
consolidado = []
for j in range(25,35):
    for i in range(10):
        model.fit([mrna_train, cnv_train, meth_train, protein_train], dsurv_train,
                  epochs=j, batch_size=10, verbose=0)
        
        y_pred = model.predict([mrna_test, cnv_test, meth_test, protein_test])
        consolidado.append(mean_squared_error(dsurv_test, y_pred))
    print("epochs: " + str(j) + " Mean-> mseTest: " + str(np.mean(consolidado)) + " Variance: " + str(np.var(consolidado)))
#print("Test MSE:", mean_squared_error(dsurv_test, y_pred))

# %%

## Multi-modal NN (mrna, cnv, meth) y protein
model = allToDNN(mrna_train, cnv_train, meth_train, protein_train, neurons=100, dropOut=.4, activation="relu")
consolidado = []
for j in range(25,35):
    for i in range(30):
        model.fit([mrna_train, cnv_train, meth_train, protein_train], dsurv_train, epochs=j, verbose=0)
        
        y_pred = model.predict([mrna_test, cnv_test, meth_test, protein_test])
        consolidado.append(mean_squared_error(dsurv_test, y_pred))
    print("epochs: " + str(j) + " Mean-> mseTest: " + str(np.mean(consolidado)) + " Variance: " + str(np.var(consolidado)))
    #print("Test MSE:", mean_squared_error(dsurv_test, y_pred))

# %%
# Correr en python 2.7
# Multiple Kernel Learning
from mklTCGA import ICDoneKernelOneTCGA, ridgeLowRankOneKernel, variousKernelVariousMethodsOneTCGA,mklTCGA

# %%

'''
Kernels:
    linear_kernel
    sigmoid_kernel
    poly_kernel:
        {"degree": 3}
    rbf_kernel:
        {"sigma": 30}
methods:
    "nystrom", "icd"
'''
from mklaren.kernel.kernel import linear_kernel, poly_kernel, sigmoid_kernel, rbf_kernel, exponential_absolute, periodic_kernel, matern_kernel
model = ICDoneKernelOneTCGA(mrna, kernel=linear_kernel, kernel_args={}, rank=15)
model = ICDoneKernelOneTCGA(mrna, kernel=poly_kernel, kernel_args={"degree": 3}, rank=15)
model = ICDoneKernelOneTCGA(mrna, kernel=sigmoid_kernel, kernel_args={}, rank=15)
model = ICDoneKernelOneTCGA(mrna, kernel=rbf_kernel, kernel_args={"sigma": 30}, rank=15)
# %%
kernels = {"linear_kernel":linear_kernel, 
           "poly_kernel":poly_kernel,
           "sigmoid_kernel":sigmoid_kernel, 
           "rbf_kernel":rbf_kernel, 
           "exponential_absolute":exponential_absolute, 
           "periodic_kernel":periodic_kernel, 
           "matern_kernel":matern_kernel
           }

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

for i in xrange(1,10,1):#frange(0, 1, .1): #0.5, 1.5,2.5:
    consolidado = []
    for ix_train, ix_test in kfoldProfiles(mrna_train, n_splits=5, seed=10):
        
        mrna_tr=mrna_train[ix_train]
        cnv_tr=cnv_train[ix_train]
        meth_tr=meth_train[ix_train]
        protein_tr=protein_train[ix_train]
        dsurv_tr=dsurv_train[ix_train]
        
#        if key == "linear_kernel":
#            kernel_args={'b': i}
#        elif key == "poly_kernel":
#            kernel_args={"degree": i}
#        elif key == "sigmoid_kernel":
#            kernel_args={'c': i}
#        elif key == "rbf_kernel":
#            kernel_args={"sigma": i}
#        elif key == "exponential_absolute":
#            kernel_args={"sigma": i}
#        elif key == "periodic_kernel":
#            kernel_args={"sigma": i}
#        else:
#            kernel_args={"nu": i}
#        
        try: 
            model = ridgeLowRankOneKernel(mrna_train, dsurv_train, kernel=poly_kernel, kernel_args={"degree": i}, rank=15, method="icd")
            yp = model.predict([np.array(mrna_test)])
            consolidado.append(mean_squared_error(dsurv_test, yp))
        except:
            pass
        
    print(" i: " + str(i) + " Mean-> iternal train: " + str(np.mean(consolidado)) + " Variance: " + str(np.var(consolidado)))
        




#model = ridgeLowRankOneKernel(mrna_train, mrna_test, dsurv_train, dsurv_test, rbf_kernel, kernel_args={"sigma": 110}, rank=15)

#model = variousKernelVariousMethodsOneTCGA(mrna_train, mrna_test, dsurv_train, dsurv_test, method="nystrom", rank=15)

# %% mkl

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump
    
#for i in xrange(15, 25, 1): 
#for i in 0.5, 1.5,2.5:
#for i in frange(.9, 1.015, .001): 
'''
default parameters:
    rank = 15 ## 
    sigmaKernel = 2.0 ## exponential_kernel == rfb_kernel
    degreeKernel = 2 ## poly_kernel
    biasKernel = 0 ## linear_kernel
    cKernel = 1 ## sigmoid_kernel
    sigmaABSKernel = 2.0 ## exponential_absolute
    sigmaPerKernel = 1 ## periodic_kernel
    nuKernel = 1.5 ## matern_kernel
    L2Ridge = 0 ## lbd
    LookAhead = 10 ## delta
'''
consolidado = []
for ix_train, ix_test in kfoldProfiles(mrna_train, n_splits=5, seed=10):
    
    mrna_tr=mrna_train[ix_train]
    #mrna_te=mrna_test[ix_test]
    cnv_tr=cnv_train[ix_train]
    #cnv_te=cnv_test[ix_test]
    meth_tr=meth_train[ix_train]
    #meth_te=meth_test[ix_test]
    protein_tr=protein_train[ix_train]
    #protein_te=protein_test[ix_test]
    dsurv_tr=dsurv_train[ix_train]
    #dsurv_te=dsurv_test[ix_test]
    
    try: 
        '''
        Parametros para PCA
        rank=17, sigmaKernel=120, degreeKernel=3,
                        biasKernel=0, cKernel=0, sigmaABSKernel=2, sigmaPerKernel=2,
                        nuKernel=1.5, L2Ridge=.991, LookAhead=40
        '''
        model = mklTCGA(mrna_tr, cnv_tr, meth_tr, protein_tr, dsurv_tr, rank=18, sigmaKernel=2, degreeKernel=2,
                        biasKernel=8, cKernel=1, sigmaABSKernel=2, sigmaPerKernel=1,
                        nuKernel=1.5, L2Ridge=.923, LookAhead=4)
        yp = model.predict([mrna_test, mrna_test, mrna_test, mrna_test, mrna_test, mrna_test, mrna_test,
                            cnv_test, cnv_test, cnv_test, cnv_test, cnv_test, cnv_test, cnv_test,
                            meth_test, meth_test, meth_test, meth_test, meth_test, meth_test, meth_test, 
                            protein_test, protein_test, protein_test, protein_test, protein_test, protein_test, protein_test])
        consolidado.append(mean_squared_error(dsurv_test, yp))
    except:
        pass
    
#print(" Mean-> iternal train: " + str(np.mean(consolidado)) + " Variance: " + str(np.var(consolidado)))
print("i: " + str(i) + " Mean-> iternal train: " + str(np.mean(consolidado)) + " Variance: " + str(np.var(consolidado)))
        #print("TestVal MSE:", mean_squared_error(dsurv_test, yp))
# %%
# Stacking (SVR)
## Revisar:
        ## A Comprehensive Guide to Ensemble Learning 
        # https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
        ## StackingClassifier 
        # https://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/#overview
        
# %%
from StackingModelTCGA import Stacking
from sklearn.svm import SVR
# %%

params = {
        "kernel": ["linear","poly","rbf","sigmoid"], 
        'C': [1, 10],'gamma': [0.001],"degree": [3,10], "coef0": [0,1]        
        }

#params = {"kernel": "poly", 
#        'C': 1,'gamma': 0.001,"degree": 3, "coef0": 1
#        }

# %%

#model1 = SVR()


test_pred1,train_pred1=Stacking(model="SVR",n_fold=10, train=pd.DataFrame(mrna_train),test=pd.DataFrame(mrna_test),y=dsurv_train.ravel(), params=params, grilla)

train_pred1=pd.DataFrame(train_pred1)
test_pred1=pd.DataFrame(test_pred1)

train_pred1.to_csv("StackingTCGA/train_pred1.csv")
test_pred1.to_csv("StackingTCGA/test_pred1.csv")
#best_params_={'C': 1, 'coef0': 1, 'degree': 3, 'gamma': 0.001, 'kernel': 'poly'}

# %%

#model2 = SVR()

test_pred2,train_pred2=Stacking(model="SVR",n_fold=10,train=pd.DataFrame(cnv_train),test=pd.DataFrame(cnv_test),y=dsurv_train.ravel(), params=params, grilla)

train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)

train_pred2.to_csv("StackingTCGA/train_pred2.csv")
test_pred2.to_csv("StackingTCGA/test_pred2.csv")
#best_params_={'C': 1, 'coef0': 0, 'degree': 3, 'gamma': 0.001, 'kernel': 'rbf'}

# %%

#model3 = SVR()

test_pred3,train_pred3=Stacking(model="SVR",n_fold=10,train=pd.DataFrame(meth_train),test=pd.DataFrame(meth_test),y=dsurv_train.ravel(), params=params, grilla)

train_pred3=pd.DataFrame(train_pred3)
test_pred3=pd.DataFrame(test_pred3)

train_pred3.to_csv("StackingTCGA/train_pred3.csv")
test_pred3.to_csv("StackingTCGA/test_pred3.csv")

# %%

#model4 = SVR()

test_pred4,train_pred4=Stacking(model="SVR",n_fold=10,train=pd.DataFrame(protein_train),test=pd.DataFrame(protein_test),y=dsurv_train.ravel(), params=params, grilla)

train_pred4=pd.DataFrame(train_pred4)
test_pred4=pd.DataFrame(test_pred4)

train_pred4.to_csv("StackingTCGA/train_pred4.csv")
test_pred4.to_csv("StackingTCGA/test_pred4.csv")
#best_params_={'C': 1, 'coef0': 0, 'degree': 3, 'gamma': 0.001, 'kernel': 'rbf'}

# %%
from sklearn.model_selection import GridSearchCV
stack = pd.concat([train_pred1, train_pred2, train_pred3, train_pred4], axis=1)
df_test = pd.concat([test_pred1, test_pred2, test_pred3, test_pred4], axis=1)

model = SVR()

params = {
        "kernel": ["linear","poly","rbf","sigmoid"], 
        'C': [1, 10],'gamma': [0.001],"degree": [3,10], "coef0": [0,1]        
        }

modelGrid = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, verbose=1)
modelGrid.fit(X=stack,y=dsurv_train.ravel())
print(modelGrid.best_params_)

y_pred=modelGrid.predict(df_test)

mean_squared_error(dsurv_test, y_pred)


# %%
from StackingModelTCGA import Stacking
from sklearn.svm import SVR
# %%

#MatrixFactorization

#NMF
from sklearn.decomposition import NMF

param_grid = {
        "n_components": [2,4,6,8,10], 
        "l1_ratio":[0,0.5,1]
        }
# %%
params = {
        "kernel": ["linear","poly","rbf","sigmoid"], 
        'C': [1, 10],'gamma': [0.001],"degree": [3,10], "coef0": [0,1]        
        }

# %%
model = NMF(n_components = 1000, init='random', random_state=0, l1_ratio = 0.5)
mrna_train_NMF = model.fit_transform(mrna_train)
mrna_test_NMF = model.fit_transform(mrna_test)
#H = model.components_
#grid = GridSearchCV(model, cv=3, n_jobs=-1, param_grid=param_grid)
#grid.fit(mrna_train, dsurv_train)
# %%
np.savetxt("NonMatrixFactorizationTCGA/mrna_train_NMF.txt", mrna_train_NMF)
np.savetxt("NonMatrixFactorizationTCGA/mrna_test_NMF.txt", mrna_test_NMF)
# %%
test_pred1,train_pred1=Stacking(model="SVR",n_fold=10, train=pd.DataFrame(mrna_train_NMF),test=pd.DataFrame(mrna_test_NMF),y=dsurv_train.ravel(), params=params, grilla=False)

train_pred1=pd.DataFrame(train_pred1)
test_pred1=pd.DataFrame(test_pred1)

train_pred1.to_csv("NonMatrixFactorizationTCGA/train_pred1.csv")
test_pred1.to_csv("NonMatrixFactorizationTCGA/test_pred1.csv")
#{'C': 1, 'coef0': 0, 'degree': 3, 'gamma': 0.001, 'kernel': 'rbf'}

# %%
model = NMF(n_components = 1000, init='random', random_state=0, l1_ratio = 0.5)
cnv_train_NMF = model.fit_transform(cnv_train)
cnv_test_NMF = model.fit_transform(cnv_test)

# %%
np.savetxt("NonMatrixFactorizationTCGA/cnv_train_NMF.txt", cnv_train_NMF)
np.savetxt("NonMatrixFactorizationTCGA/cnv_test_NMF.txt", cnv_test_NMF)

# %%
test_pred2,train_pred2=Stacking(model="SVR",n_fold=10, train=pd.DataFrame(cnv_train_NMF),test=pd.DataFrame(cnv_test_NMF),y=dsurv_train.ravel(), params=params, grilla=True)

train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)

train_pred2.to_csv("NonMatrixFactorizationTCGA/train_pred2.csv")
test_pred2.to_csv("NonMatrixFactorizationTCGA/test_pred2.csv")
#{'C': 1, 'coef0': 1, 'degree': 3, 'gamma': 0.001, 'kernel': 'poly'}

# %%
model = NMF(n_components = 1000, init='random', random_state=0, l1_ratio = 0.5)
meth_train_NMF = model.fit_transform(meth_train)
meth_test_NMF = model.fit_transform(meth_test)

# %%
np.savetxt("NonMatrixFactorizationTCGA/meth_train_NMF.txt", meth_train_NMF)
np.savetxt("NonMatrixFactorizationTCGA/meth_test_NMF.txt", meth_test_NMF)

# %%

meth_train_NMF = np.loadtxt("NonMatrixFactorizationTCGA/meth_train_NMF.txt")
meth_test_NMF = np.loadtxt("NonMatrixFactorizationTCGA/meth_test_NMF.txt")

params={}
test_pred3,train_pred3=Stacking(model="SVR",n_fold=10, train=pd.DataFrame(meth_train_NMF),test=pd.DataFrame(meth_test_NMF),y=dsurv_train.ravel(), params=params, grilla=False)

train_pred3=pd.DataFrame(train_pred3)
test_pred3=pd.DataFrame(test_pred3)

train_pred3.to_csv("NonMatrixFactorizationTCGA/train_pred3.csv")
test_pred3.to_csv("NonMatrixFactorizationTCGA/test_pred3.csv")
#{'C': 1, 'coef0': 1, 'degree': 3, 'gamma': 0.001, 'kernel': 'poly'}

# %%
model = NMF(n_components = 75, init='random', random_state=0, l1_ratio = 0.5)
prot_train_NMF = model.fit_transform(protein_train)
prot_test_NMF = model.fit_transform(protein_test)

# %%
np.savetxt("NonMatrixFactorizationTCGA/prot_train_NMF.txt", prot_train_NMF)
np.savetxt("NonMatrixFactorizationTCGA/prot_test_NMF.txt", prot_test_NMF)

# %%
test_pred4,train_pred4=Stacking(model="SVR",n_fold=10, train=pd.DataFrame(prot_train_NMF),test=pd.DataFrame(prot_test_NMF),y=dsurv_train.ravel(), params=params, grilla=True)

train_pred4=pd.DataFrame(train_pred4)
test_pred4=pd.DataFrame(test_pred4)

train_pred4.to_csv("NonMatrixFactorizationTCGA/train_pred4.csv")
test_pred4.to_csv("NonMatrixFactorizationTCGA/test_pred4.csv")

# %%
from sklearn.model_selection import GridSearchCV
train_pred1 = pd.read_csv("NonMatrixFactorizationTCGA/train_pred1.csv", header=0)
test_pred1 = pd.read_csv("NonMatrixFactorizationTCGA/test_pred1.csv", header=0)
train_pred2 = pd.read_csv("NonMatrixFactorizationTCGA/train_pred2.csv", header=0)
test_pred2 = pd.read_csv("NonMatrixFactorizationTCGA/test_pred2.csv", header=0)
train_pred3 = pd.read_csv("NonMatrixFactorizationTCGA/train_pred3.csv", header=0)
test_pred3 = pd.read_csv("NonMatrixFactorizationTCGA/test_pred3.csv", header=0)
train_pred4 = pd.read_csv("NonMatrixFactorizationTCGA/train_pred4.csv", header=0)
test_pred4 = pd.read_csv("NonMatrixFactorizationTCGA/test_pred4.csv", header=0)

stack = pd.concat([train_pred1, train_pred2, train_pred3, train_pred4], axis=1)
df_test = pd.concat([test_pred1, test_pred2, test_pred3, test_pred4], axis=1)

model = SVR()

params = {
        "kernel": ["linear","poly","rbf","sigmoid"], 
        'C': [1, 10],'gamma': [0.001],"degree": [3,10], "coef0": [0,1]        
        }

modelGrid = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, verbose=1)
modelGrid.fit(X=stack,y=dsurv_train.ravel())
print(modelGrid.best_params_)

y_pred=modelGrid.predict(df_test)

mean_squared_error(dsurv_test, y_pred)

# %%
#svmLinearParam = {
#        "kernel":"linear", 
#        "C":1, 
#        "epsilon": 0.1
#        }
#svmPolyParam = {
#        "kernel":"poly", 
#        "C":1, 
#        "degree": 3, 
#        "gamma": "auto",
#        "coef0":0
#        }
#svmRbfParam = {
#        "kernel":"rbf", 
#        "C":1, 
#        "epsilon": 0.1
#        }
#svmSigmoidParam = {
#        "kernel":"sigmoid", 
#        "gamma": "auto",
#        "coef0": 0
#        }
## %%
#from mlxtend.regressor import StackingCVRegressor
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import GridSearchCV
#
#svrLinear = SVR(kernel="linear")
#svrPoly = SVR(kernel="poly")
#svrRbf = SVR(kernel="rbf")
#svrSigmoid = SVR(kernel="sigmoid")
#rf = RandomForestRegressor(random_state=10)
#sclf = StackingCVRegressor(regressors=[svrLinear, svrPoly, svrSigmoid], 
#                          meta_regressor=rf)
#
#params = {'SVR__C': [1, 10],'SVR__gamma': [0.001, 0.0001],"SVR__degree": [3,10,15],
#          "SVR__epsilon": [0.01, 0.1], "SVR__coef0": [0,1],
#          'meta-randomforestregressor__n_estimators': [10, 100]}
#
#grid = GridSearchCV(estimator=sclf, 
#                    param_grid=params, 
#                    cv=3,
#                    refit=True)
#
##svmLinear = SklearnHelper(model=SVR,params=svmLinearParam)
##svmPoly = SklearnHelper(model=SVR, params=svmPolyParam)
##svmRbf = SklearnHelper(model=SVR, params=svmRbfParam)
##svmSigmoid = SklearnHelper(model=SVR, params=svmSigmoidParam)
#
## %%
#grid.fit(mrna_train, dsurv_train)