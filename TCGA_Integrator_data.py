# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:49:20 2019

@author: da.salazarb
"""

## https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf
# %% Librerias
import re
import numpy as np
import pandas as pd
import scipy.io as sio
from softImpute import SoftImpute
from sklearn.preprocessing import MinMaxScaler

# %% Limpiar datos

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

#log2-pseudo 1
def logPseudo(tcga):
    tcga = tcga.transform(lambda x: np.log2(x + 1))
    return tcga

# Seleccion de variables para varianzas pequeñas
def varianceSelection(X, THRESHOLD = 10):
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=THRESHOLD)
    sel.fit_transform(X)
    return X[[c for (s, c) in zip(sel.get_support(), X.columns.values) if s]]

#sofImpute
def imputeSoft(tcga):
    data = tcga.values
    clf = SoftImpute()
    clf.fit(data)
    data = clf.predict(data)
    tcga = pd.DataFrame(data, index=tcga.index, columns=tcga.columns)
    return tcga

# Obtener train y dev set de cada profile
def train_test_profile(tcga):
    #from random import seed
    #from random import sample
    sequence = [i for i in range(tcga[list(tcga.keys())[0]].shape[0])]
    
    np.random.shuffle(sequence)
    train, test = sequence[:int(len(sequence)*.80)], sequence[int(len(sequence)*.80):]
    
    tcga_train = {k: v.iloc[train,:] for k, v in tcga.items()}
    tcga_test = {k: v.iloc[test,:] for k, v in tcga.items()}
    
    return tcga_train, tcga_test

## random variables x 100
def random_variables(tcga, n_rand_var):
    sequence = {k: list(range(0,len(v.columns))) for k, v in tcga.items()}
    [np.random.shuffle(seq) for seq in sequence.values()]
    tcga = {key: value.iloc[:,sequence[key][0:n_rand_var]] for key, value in tcga.items() if key != "lgg_proteinTCGA" or key != "lgg_survivalTCGA"}
    return tcga

# %%
class MajorProfile:
    #Data.Frame del archivo .DaTa obtenido de TCGA-Integrator
    # Del .mat se extraen la matriz -> "Features", los nombres de las filas -> "Symbols y de las columnas -> "Samples"
    # Por otro lado se obtiene survival (la variable respuesta)
    ## constructor
    
    # Segun __[TCGAIntegrator](https://github.com/cooperlab/TCGAIntegrator)__, las plataformas se diferencian por el sufijo: 
    # ('_Clinical', '_Mut', '_CNV', '_CNVArm', '_Protein' and '_mRNA')
    def __init__(self, _PathProfile, _typeCancer=""):
        self.profile = sio.loadmat(_PathProfile, struct_as_record=False, squeeze_me=True)
        self.typeCancer = _typeCancer
        
    def to_dataFrame(self):
        justmRNA = np.where((self.profile["AvailablemRNA"] == "Yes") & (list(map(lambda x: re.search("-02$",x) == None, self.profile["Samples"]))))
        symbols = self.profile["Symbols"]
        survival = pd.DataFrame(self.profile["Survival"], index=self.profile["Samples"], columns=["Survival"])
        _profile = pd.DataFrame(self.profile["Features"], index=symbols, columns=self.profile["Samples"])
        
        #concatenacion, transposicion y limpieza de nombres
        _profile = pd.concat([_profile, survival.transpose()])
        _profile = _profile.transpose()
        
        #los nombres de las columnas vienen con muchos espacios, se deben eliminar
        _profile = _profile.rename(columns=lambda x: x.strip()) 
        
        _profile = _profile.iloc[justmRNA[0]]
        
        return _profile
    
    def get_list_genes(self):
        return pd.Series(self.profile["Symbols"])[pd.Series(self.profile["Symbols"]).str.contains("_mRNA")].str.strip().str.replace("_mRNA", "", regex=False).tolist()
    
    def __str__(self):
        return("{} {}".format("We are working with: ", self.typeCancer) + ". " + 
               "{} {}".format("The original shape is: ", self.profile["Features"].shape))
        
# %%
class Clean:
    # Eliminar variables con varianza cero
    def varNearZero(cls, tcga, Threshold, _sufijo):
        print("----------------------------")
        print("Removiendo varianza de " + _sufijo)
        print("----------------------------")
        tcga=tcga.filter(regex=_sufijo) #filtar por plataforma
        print("old.shape " + _sufijo + ": " + str(tcga.shape))
        is_noise=tcga.var(skipna =True) >= Threshold #encontrar variables con varianzas mayores a noise
        tcga=tcga.loc[:,is_noise] #escoger variables filtradas
        #min_max_scaler = preprocessing.MinMaxScaler()
        #tcga_x = min_max_scaler.fit_transform(tcga_x)
        #tcga_x = pd.DataFrame(tcga_x, index=tcga.index, columns=tcga.columns)
        print("new.shape " + _sufijo + ": " + str(tcga.shape))
        return tcga
        
    # Verificacion de datos faltantes por grupo de variables
    def verify_nan(cls, tcga, _sufijo, clean, method):
        #tcga=tcga.filter(regex=regex)
        #Existen o no NaN
        print("----------------------------")
        print("Verificacion de faltantes de " + _sufijo)
        print("----------------------------")
        try:
            tcga=tcga.filter(regex=_sufijo) #filtar por plataforma
        except:
            _sufijo = "Una omica"
        if tcga.isnull().values.any() == False:
            print("¿Existen valores NaN? ", tcga.isnull().values.any())
        else:
            print("Para %s existen %s NaNs" % (_sufijo, tcga.isnull().sum().sum()))
        #Se limpian o no
        if clean == True and tcga.isnull().values.any() == True:
            if type(method) == int:
                tcga=tcga.fillna(method) #{‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}
                print("Datos limpios para " + _sufijo)
            else:
                tcga=tcga.fillna(method=method)
        else:
            print("Los valores clean y method no se usaron. No hay NaN.")
        if tcga.isnull().sum().sum() > 0:
            print("**No se logro imputar datos**")
        return tcga
    
    def mRNAconsensus(cls, mRNA, Threshold, clean, metodo):
        """ Escoge los pacientes que tienen el perfil completo de mRNA
        Inputs:
            mRNA: perfil de mrna
            Threshold: varianza aceptada para eliminar 
        Outputs:
            Escoge pacientes con todos los perfiles
        """
        print("----------------------------")
        print("Consenso de mRNA")
        print("----------------------------")
        
        mRNA = cls.verify_nan(mRNA, _sufijo="_mRNA", clean=clean, method=metodo)
        mRNA_new=cls.varNearZero(mRNA, Threshold, _sufijo="_mRNA") #varianza cero
        pac_mRNA=mRNA_new.T #trasponer  -> pacientes como columnas
        is_noise=pac_mRNA.var() >= Threshold #ej: 0.1. Valor minimo de varianza aceptada
        pac_mRNA_nearVarZero=pac_mRNA.loc[:,is_noise] #nuevo dataFrame con variables con varianza diferente a cero
        print("new.shape.mrna: " + str(pac_mRNA_nearVarZero.shape))
        mRNA_new=pac_mRNA_nearVarZero.T #retorno de pacientes como filas
        return mRNA_new
    
    def toNonNegative(cls, tcga):
        scaler = MinMaxScaler(copy=False, feature_range=(0, 1))
        scaler.fit_transform(tcga)
        return tcga

# %%
class Data(Clean):
    
    def __init__(self, data):
        try:
            self.data = data.dropna(axis=1, how='all')
        except:
            pass
        
    def update_df(self, data):
        self.data = data
        
    def mutate(self, mutation, *args):
        data = mutation(self.data, *args)
        self.update_df(data)
        return data
    
    def updateColumnNames(self, start, end):
        self.data.columns = list(map(lambda x: start + str.replace(x, end, ""), self.data.columns))
    
    def saveListVar(self,_regex, path):
        genLista = list(map(lambda x: str.replace(x, _regex, ""), self.data))
        
        print(", ".join(genLista))

        with open(path+"/"+str.replace(_regex,"_","")+"_Lista.txt", "w") as output:
            output.write("" + ", ".join(map(str, genLista)) + "")

# %%
        
class MethData(Data):
    
    def __init__(self, pathMeth, fileMeth):
        self.data = pd.read_csv(pathMeth+fileMeth, index_col=0, sep="\t", dtype=str)
    
    # Lectura del perfil de metilacion
    def methConsensus(self):
        print("----------------------------")
        print("Consenso de DNAMeth ")
        print("----------------------------")
        ## Lectura e imputacion.
         #index_col -> primera columna como nombres de filas
        self.data = self.data.drop(self.data.index[0]) #eliminar la primera fila que contiene el valor "Beta_Value" en todas las casillas
        #self.data = self.data.fillna(method="ffill") #imputacion para datos faltantes
        
        self.data = self.data.T # formato columnas: genes - filas: pacientes
        self.data = self.data.rename(columns=lambda x: "met_" + x) # renombrar las variables ej: meth_A1BG
        self.data.columns.name = "" # quitar "Hybridization REF"
        ## Concatenacion de perfil de metilacion con los demas
        #self.dnaMet = pd.concat([self.dnaMet, mRNA_new], axis=1)
        #self.data = pd.concat([self.data, mRNA_new], axis=1, sort=False)
        #self.data = self.data.loc[list(mRNA_new.index)]
        #self.data = self.data.verify_nan(self.data, _sufijo="_Met", clean=True, method="ffill")
        self.data = self.data.apply(pd.to_numeric)
        return self.data

# %%
class AssemblerData(Data):
    
    def get_patientsAssembler(self, samples):
        try:
            self.data = self.data[self.data["Strand"] == "+"]
        except:
            pass
        self.data.columns = list(map(lambda x: x[0:15], self.data.columns))
        self.data.set_index("GeneSymbol", inplace = True)
        self.data = self.data[list(set(samples).intersection(set(list(map(lambda x: x[0:15], self.data.columns)))))]
        self.data = self.data.loc[:,~self.data.columns.duplicated()]
        self.data = self.data.T
        self.data = self.data.dropna(axis=1, how='all')

# %% The Cancer Prtoein Atlas - uso exclusivo del protein array
class TCPAData(Data):
    
    def get_patientTCPA(self, samples):
        #self.data.index = self.data.index.str.extract("([a-zA-Z0-9]+\s*)", expand=False)
        self.data.columns = list(map(lambda x: x[0:15], self.data.columns))
        self.data = self.data[list(set(samples).intersection(set(self.data.columns)))]
        self.data = self.data.T
        self.data.columns = list(map(lambda x: x.replace("|", "_-_"), self.data.columns))
        
# %% mirna 
class MirnaData(Data):
    
    def __init__(self, pathMeth, fileMeth):
        self.data = pd.read_csv(pathMeth+fileMeth, index_col=0, sep="\t", dtype=str)
    
    # modificacion de mirna
    def mirnaConsensus(self):
        print("----------------------------")
        print("Consenso de miRNA ")
        print("----------------------------")
        ## Lectura e imputacion.
         #index_col -> primera columna como nombres de filas
        #self.data = self.data.drop(self.data.index[0]) #eliminar la primera fila que contiene el valor "Beta_Value" en todas las casillas
        #self.data = self.data.fillna(method="ffill") #imputacion para datos faltantes
        self.data.columns = list(map(lambda x: x[0:15], self.data.columns))
        self.data = self.data.T # formato columnas: genes - filas: pacientes
        self.data = self.data.rename(columns=lambda x: "mirna_" + x) # renombrar las variables ej: meth_A1BG
        self.data.columns.name = "" # quitar "Hybridization REF"
        ## Concatenacion de perfil de metilacion con los demas
        #self.dnaMet = pd.concat([self.dnaMet, mRNA_new], axis=1)
        #self.data = pd.concat([self.data, mRNA_new], axis=1, sort=False)
        #self.data = self.data.loc[list(mRNA_new.index)]
        #self.data = self.data.verify_nan(self.data, _sufijo="_Met", clean=True, method="ffill")
        return self.data
    
# %%
class NormalTissue(Data):
    
    def __init__(self, pathMeth, fileMeth):
        self.data=pd.read_csv(pathMeth+fileMeth, index_col=0, sep="\t", dtype=str)
        
    def normalTissueConsensus(self, tcga):
        print("----------------------------")
        print("Buscando Tejido sano -11R$ ")
        print("----------------------------")
        ## dejar nombres cortos para samples
        self.data.columns = list(map(lambda x: x[0:15], self.data.columns))
        ## buscar los terminados en 11 -> normal sample
        justNormalTissue = np.where(list(map(lambda x: re.search("-1[0-9]$",x), self.data.columns)))
        ## escoger normal sample
        self.data = self.data.iloc[:,justNormalTissue[0]]
        ## adicionar mrna_ a los genes
        self.data.index = "mrna_"+self.data.index
        ## buscar los genes que tiene mrna.data.columns en este data frame
        self.data = self.data.loc[set(tcga.data.columns.intersection(self.data.index)),:]
        ## hay duplicados solo quedarse con el ultimo
        self.data = self.data[~self.data.index.duplicated(keep='last')]
        ## transponer para maentener mismo formato
        self.data = self.data.T
        ## devolver data
        return self.data
    

# In[29]:
# #KFold para obtener los train y test de cada perfil
# def kfoldProfiles(tcga, n_splits, seed): 
#     from sklearn.model_selection import KFold
#     kfold = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
#     splits = kfold.split(tcga)
#     return splits

# In[30]:

# #Para cargar nuevamente los archivos de forma rapida
# def reRunFiles(norm=True, varX=False, logSurv=False, varianza=False, randVar=(False, 0), pcaVar=(False, 0), preVitaVar=False, setTrainTest=False, preMRmrVar=False):
#     '''
#     varX: True para eliminar la primera variable que puede ser el identificador del paciente
#     logSurv: True para sacar logaritmo de 'y' o dsurv
#     varianza: True para eliminar variables por varianza segun threshold
#     randVar: True para elegir variables, int para escoger una cantidad de variables
#     PreVar: True para elegir 100 variables aleatorias predefinidas con poca correlacion entre ellas
#     setTrainTest:  True para retornar train y dev set
    
#     '''
#     mrna=pd.read_table("TCGA-Integrator/mrnaPrePros.txt")
#     cnv=pd.read_table("TCGA-Integrator/cnvPrePros.txt")
#     meth=pd.read_table("TCGA-Integrator/methPrePros.txt")
#     dsurv=pd.read_table("TCGA-Integrator/dsurvPrePros.txt")
#     protein=pd.read_table("TCGA-Integrator/dataTCGA/originalData/new_gbm_Protein.txt", sep=",", index_col=0)
    
#     # Se elimina la var X
#     if varX == True:
#         mrna=mrna.iloc[:,1:]
#         cnv=cnv.iloc[:,1:]
#         meth=meth.iloc[:,1:]
#         dsurv=dsurv.iloc[:,1:]
    
#     if norm == True:
#         # Normalizar
#         mrna=normalizar(mrna)
#         cnv=normalizar(cnv)
#         meth=normalizar(meth)
#         protein=normalizar(protein)
#     else:
#         # NormalizarRango
#         mrna=scalarRango(mrna)
#         cnv=scalarRango(cnv)
#         meth=scalarRango(meth)
#         protein=scalarRango(protein)
    
#     # Los archivos vienen con una columna "X" que contiene los indices de los pacientes
#     # si varX == True -> se elimina esa primera columna
    
# #    if preVar == True:
# #        #Lista de 100 variables aleatoria
# #        mrnaLista=['JPH1', 'MT1X', 'NR2E3', 'PEX5L', 'CHST6', 'VEGFC', 'TPSD1', 'TMUB1', 'ACER1', 'OXER1', 'VAV3', 'HAND2', 'CYP11A1', 'HS3ST3B1', 'HIF3A', 'RPL21', 'LDLR', 'MUC15', 'PON2', 'WDR72', 'SH2D4B', 'SLC7A2', 'S100A3', 'RCVRN', 'USP29', 'AQP2', 'FAM83H', 'SPAG4', 'TP53I11', 'SORCS3', 'GLT1D1', 'CCIN', 'TFAP2B', 'GAD1', 'EPHX3', 'KRT17', 'GUCY2D', 'DIRAS2', 'HIST1H2AB']
# #        mrna=mrna[mrnaLista]
# #        
# #        #Lista de 100 variables aleatoria
# #        cnvLista=['FAM8A1_CHR6_pos_cnv', 'LSP1P3_CHR5_pos_cnv', 'CDCP1_CHR3_neg_cnv', 'RPS14_CHR5_neg_cnv', 'THOC2_CHRX_neg_cnv', 'GPR6_CHR6_pos_cnv', 'LRRC10B_CHR11_pos_cnv', 'TSSK3_CHR1_pos_cnv', 'ATP6V1A_CHR3_pos_cnv', 'PLGLB2_CHR2_pos_cnv', 'SNORD68_CHR16_pos_cnv', 'TRIM66_CHR11_neg_cnv', 'UBE2N_CHR12_neg_cnv', 'TAOK1_CHR17_pos_cnv', 'ARHGAP10_CHR4_pos_cnv', 'CCDC9_CHR19_pos_cnv', 'IFNA13_CHR9_neg_cnv', 'DOK7_CHR4_pos_cnv', 'RBM44_CHR2_pos_cnv', 'SNRNP48_CHR6_pos_cnv', 'B3GNT3_CHR19_pos_cnv', 'ALOX5AP_CHR13_pos_cnv', 'ATRX_CHRX_neg_cnv', 'LOC643650_CHR10_neg_cnv', 'SREBF2_CHR22_pos_cnv', 'SNAI1_CHR20_pos_cnv', 'SHC3_CHR9_neg_cnv', 'CBX3P2_CHR18_neg_cnv', 'GOLGA6C_CHR15_pos_cnv', 'MAGEB1_CHRX_pos_cnv', 'ARHGAP42_CHR11_pos_cnv', 'KCNK9_CHR8_neg_cnv', 'SOS1_CHR2_neg_cnv', 'SERPINA10_CHR14_neg_cnv', 'CLEC18A_CHR16_pos_cnv', 'PDCD1LG2_CHR9_pos_cnv', 'ROR1_CHR1_pos_cnv', 'APOBEC1_CHR12_neg_cnv', 'PRKD1_CHR14_neg_cnv', 'RNF41_CHR12_neg_cnv', 'COX6A2_CHR16_neg_cnv', 'RTL8B_CHRX_neg_cnv', 'MINDY1_CHR1_neg_cnv', 'F2RL2_CHR5_neg_cnv', 'LOXL2_CHR8_neg_cnv', 'LOC100498859_CHR3_pos_cnv', 'GK2_CHR4_neg_cnv', 'SNORD1A_CHR17_pos_cnv', 'LGSN_CHR6_neg_cnv', 'POLI_CHR18_pos_cnv', 'CD24_CHRY_neg_cnv', 'KLHL11_CHR17_neg_cnv']
# #        cnv=cnv[cnvLista]
# #        
# #        #Lista de 100 variables aleatoria
# #        methLista=['IL6_Met', 'KCNF1_Met', 'OSTBETA_Met', 'ITGA8_Met', 'ALOX15_Met', 'CCNL1_Met', 'CCM2_Met', 'CXCL1_Met', 'KCNH8_Met', 'RASD2_Met', 'TMEM49_Met', 'HLA.A_Met']
# #        meth=meth[methLista]
# #        
# #        #Lista de 100 variables aleatoria
# #        proteinLista=['XRCC5_Protein', 'FASN_Protein', 'CDKN1A_Protein', 'SMAD1_Protein', 'TFRC_Protein', 'BCL2_Protein', 'GAPDH_Protein', 'SYK_Protein', 'CDKN1B_Protein', 'NFKB1_Protein', 'PDCD4_Protein', 'RPS6_Protein.2', 'ERBB2_Protein', 'PCNA_Protein', 'MAPK14_Protein.1', 'CDK1_Protein', 'LCK_Protein', 'MYH11_Protein', 'G6PD_Protein', 'ASNS_Protein', 'AXL_Protein', 'AKT1 AKT2 AKT3_Protein.1', 'BECN1_Protein', 'ITGA2_Protein', 'PRKAA1_Protein', 'BRD4_Protein', 'CAV1_Protein', 'IRF1_Protein', 'SETD2_Protein', 'EGFR_Protein.1', 'CDH3_Protein', 'MAP2K1_Protein', 'SRC_Protein', 'MYH9_Protein']
# #        protein=protein[proteinLista]
    
#     if preVitaVar == True:
#         vitaVar=pd.read_table("C:/Users/da.salazarb/Google Drive/Tutorial_2018-10/03_Resultados/DataTCGA/varSelected/vita/vitaVarSelect.txt")
#         mrna=mrna[list(vitaVar.iloc[:,0])]
#         cnvNames = []
#         for i in list(vitaVar.iloc[:,0]):
#             [cnvNames.append(s) for s in cnv.columns if i in s]
#         cnv=cnv[cnvNames]
#         methNames = []
#         for i in list(vitaVar.iloc[:,0]):
#             [methNames.append(s) for s in meth.columns if i in s]
#         meth=meth[methNames]
        
#     if preMRmrVar == True:
#         MRmrVar=pd.read_table("C:/Users/da.salazarb/Google Drive/Tutorial_2018-10/03_Resultados/DataTCGA/varSelected/mrMR/mrMRoneResultVarSelect.txt")
#         mrna=mrna[list(MRmrVar.iloc[:,0])]
#         cnvNames = []
#         for i in list(MRmrVar.iloc[:,0]):
#             [cnvNames.append(s) for s in cnv.columns if i in s]
#         cnv=cnv[cnvNames]
#         methNames = []
#         for i in list(MRmrVar.iloc[:,0]):
#             [methNames.append(s) for s in meth.columns if i in s]
#         meth=meth[methNames]
    
#     # Logaritmo a dsurv
#     if logSurv == True:
#         dsurv=np.log10(dsurv)
    
#     # Correccion de nombres para protein
#     nombreProteinIndex=list(protein.index)
#     nombreProteinIndex = [w.replace('-', '.') for w in nombreProteinIndex]
#     protein.index=nombreProteinIndex
#     protein=protein.loc[mrna.index,:]
    
#     # Seleccionar por varianza
#     if varianza == True:
#         mrna=varianceSelection(mrna, THRESHOLD=0.5)
#         cnv=varianceSelection(cnv, THRESHOLD=0.01)
#         meth=varianceSelection(meth, THRESHOLD=0.001)
#         protein=varianceSelection(protein, THRESHOLD=0.1)
    
#     # variables aleatorias
#     if randVar[0] == True:
#         variables=np.random.choice(len(mrna.columns), randVar[1])
#         mrna=mrna.iloc[:,variables]
#         variables=np.random.choice(len(cnv.columns), randVar[1])
#         cnv=cnv.iloc[:,variables]
#         variables=np.random.choice(len(meth.columns), randVar[1])
#         meth=meth.iloc[:,variables]
#         variables=np.random.choice(len(protein.columns), randVar[1])
#         protein=protein.iloc[:,variables]
    
#     # PCA
#     if pcaVar[0] == True:
#         from sklearn.decomposition import PCA
#         pca = PCA(n_components=pcaVar[1])
#         mrna = pca.fit_transform(mrna)
#         cnv = pca.fit_transform(cnv)
#         meth = pca.fit_transform(meth)
#         protein = pca.fit_transform(protein)
        
#     if setTrainTest == True:
#         # Obtener train y dev set de cada profile
#         from sklearn.model_selection import train_test_split
#         mrna_train, mrna_test, dsurv_train, dsurv_test = train_test_split(pd.DataFrame(mrna), pd.DataFrame(dsurv), test_size=.2, random_state=10)
#         cnv_train=pd.DataFrame(cnv).loc[mrna_train.index]
#         cnv_test=pd.DataFrame(cnv).loc[mrna_test.index]
#         meth_train=pd.DataFrame(meth).loc[mrna_train.index]
#         meth_test=pd.DataFrame(meth).loc[mrna_test.index]
#         protein_train=pd.DataFrame(protein).loc[mrna_train.index]
#         protein_test=pd.DataFrame(protein).loc[mrna_test.index]
        
#         mrna_train_index = mrna_train.index
#         mrna_train_columns = mrna_train.columns
#         mrna_test_index = mrna_test.index
#         mrna_test_columns = mrna_test.columns
#         cnv_train_index = cnv_train.index
#         cnv_train_columns = cnv_train.columns
#         cnv_test_index = cnv_test.index
#         cnv_test_columns = cnv_test.columns
#         meth_train_index = meth_train.index
#         meth_train_columns = meth_train.columns
#         meth_test_index = meth_test.index
#         meth_test_columns = meth_test.columns
#         protein_train_index = protein_train.index
#         protein_train_columns = protein_train.columns
#         protein_test_index = protein_test.index
#         protein_test_columns = protein_test.columns
#         dsurv_train_index = dsurv_train.index
#         dsurv_train_columns = dsurv_train.columns
#         dsurv_test_index = dsurv_test.index
#         dsurv_test_columns = dsurv_test.columns
        
#         print("Dimensiones mrna: {0}".format(mrna.shape))
#         print("Dimensiones cnv: {0}".format(cnv.shape))
#         print("Dimensiones meth: {0}".format(meth.shape))
#         print("Dimensiones dsurv: {0}".format(dsurv.shape))
#         print("Dimensiones protein: {0}".format(protein.shape))
        
#         # Normalizar
#         mrna=np.array(mrna)
#         cnv=np.array(cnv)
#         meth=np.array(meth)
#         protein=np.array(protein)
#         dsurv=np.array(dsurv)
#         mrna_train=np.array(mrna_train)
#         mrna_test=np.array(mrna_test)
#         cnv_train=np.array(cnv_train)
#         cnv_test=np.array(cnv_test)
#         meth_train=np.array(meth_train)
#         meth_test=np.array(meth_test)
#         protein_train=np.array(protein_train)
#         protein_test=np.array(protein_test)
#         dsurv_train=np.array(dsurv_train)
#         dsurv_test=np.array(dsurv_test)
        
#         return mrna, cnv, meth, dsurv, protein, mrna_train, mrna_test, cnv_train, cnv_test, meth_train, meth_test, protein_train, protein_test, dsurv_train, dsurv_test, mrna_train_index, mrna_train_columns, mrna_test_index, mrna_test_columns, cnv_train_index, cnv_train_columns, cnv_test_index, cnv_test_columns, meth_train_index, meth_train_columns, meth_test_index, meth_test_columns, protein_train_index, protein_train_columns, protein_test_index, protein_test_columns, dsurv_train_index, dsurv_train_columns, dsurv_test_index, dsurv_test_columns
#     else:
#         print("Dimensiones mrna: {0}".format(mrna.shape))
#         print("Dimensiones cnv: {0}".format(cnv.shape))
#         print("Dimensiones meth: {0}".format(meth.shape))
#         print("Dimensiones dsurv: {0}".format(dsurv.shape))
#         print("Dimensiones protein: {0}".format(protein.shape))
        
#         # Normalizar
#         mrna=np.array(mrna)
#         cnv=np.array(cnv)
#         meth=np.array(meth)
#         protein=np.array(protein)
#         dsurv=np.array(dsurv)
        
#         return mrna, cnv, meth, dsurv, protein