# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:01:57 2019

@author: da.salazarb
"""
import os
import scipy.io as sio
#import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import normalize
from TCGA_Integrator_data import *

# TCGA-Integrator

# Los datos se obtuvieron de __[FireHose Brad GDAC](https://gdac.broadinstitute.org/)__. 
# Se descargaron los datos de LGG y GBM empleando la plataforma __[TCGAIntegrator](https://github.com/cooperlab/TCGAIntegrator)__. 
# Las instrucciones de como emplear este paquete se encuentran en la pagina de GitHub, pero para mayor aplicabilidad guiarse con el archivo 
# **GuiaTCGAintegrator.py** ubicado en la ruta Tutorial -> 03 Resultados -> DataTCGA.

# Los parametros usados en TCGAintegrator fueron *MuSigQ* y *GisticQ* de 0.1 (ver GuiaTCGAintegrator.py). <br> <br>
# Recordar que desde esta plataforma solo se puede descargar datos clinicos, CNV, CNA, Mutaciones, mRNA, proteinas y miRNA.
# Para el perfil de metilacion se debe ir directamente a la pagina de __[FireBrowse](http://firebrowse.org/)__, seleccionar el cohorte, 
# aqui aparece un grafico de barras, se debe seleccionar la barra de *methylation* y descargar el archivo *Methylation_Preprocess (MD5)*. <br> <br>
# Iniciare con la lectura y preprocesamiento de los datos de cada conjunto de datos, luego seguire con el perfil de metilacion.
# Finalmente, unire las dos bases de datos.
def loadTCGA(path):
    # %%
    #path = "C:/Users/da.salazarb/Google Drive/Tutorial_2018-10/03_Resultados/DataTCGA/DeepTCGA"
    #path = "C:/Users/AQ01092/Downloads/DeepTCGA"
    ## Ubicacion de archivo LGG.Data
    #LGG = MajorProfile("C:/Users/da.salazarb/Google Drive/Tutorial_2018-10/03_Resultados/DataTCGA/TCGA-Integrator/LGG.Data", "LGG")
    LGG = MajorProfile(path+"/LGG.Data", "LGG")
    lgg = LGG.to_dataFrame() # Este dataset proviene de Integrator. De aqui puede salir mrna, protein, cnv, mut.
    
    # %% mrna - la funcion mRNAconsensus quita genes con poca varianza
    mrna = Data(lgg)
    mutate = mrna.mutate
    mutate(mrna.mRNAconsensus, 30, False, 1) # no se imputan datos aun porque se hara con SoftImpute en R
    #mutate(mrna.toNonNegative)
    mrna.updateColumnNames("mrna_", "_mRNA")
    
    # %% meth - descargado directamente de Firehose Broad GDAC (https://gdac.broadinstitute.org/)
    #pathMeth = "C:/Users/da.salazarb/Google Drive/Tutorial_2018-10/03_Resultados/DataTCGA/TCGA-Integrator" # ruta del perfil DNA methyl 
    fileMeth = "/LGG.meth.by_mean.data.txt"
    #pathMeth = "D:\LGG_GBM"; fileMeth = "/LGG.meth.by_mean.data.txt"
    meth = MethData(path, fileMeth)
    meth.methConsensus()
    mutate = meth.mutate
    mutate(meth.verify_nan, "", False, 1) # no se imputan datos aun porque se hara con SoftImpute en R
    #mutate(meth.varNearZero, 0.01, "met_") # no se imputan datos aun porque se hara con SoftImpute en R
    
    # %% cnv - dataset proviene de assembler
    #path = "C:/Users/da.salazarb/Desktop/TCGA-Assembler/QuickStartExample/Part2_BasicDataProcessingResult"
    cnv = AssemblerData(pd.read_csv(path+"/LGG__copyNumber.txt", sep="\t"))
    cnv.get_patientsAssembler(samples=mrna.data.index)
    mutate = cnv.mutate
    mutate(cnv.verify_nan, "", False, 1) # no se imputan datos aun porque se hara con SoftImpute en R
    #mutate(cnv.toNonNegative)
    cnv.updateColumnNames("cnv_", "")
    
    # %% protein - descargado directamente de Firehose Broad GDAC (https://gdac.broadinstitute.org/)
    #path = "C:/Users/da.salazarb/Google Drive/Tutorial_2018-10/03_Resultados/DataTCGA/TCGA-Integrator"
    protein = TCPAData(pd.read_csv(path+"/LGG.rppa.txt", index_col=0, sep="\t"))
    protein.get_patientTCPA(samples=mrna.data.index)
    mutate = protein.mutate
    mutate(protein.verify_nan, "", False, 1) # no se imputan datos aun porque se hara con SoftImpute en R
    protein.updateColumnNames("protein_", "")
    
    # %%
    tcga = {"lgg_cnvTCGA": cnv.data,
            "lgg_mrnaTCGA": mrna.data,
            "lgg_proteinTCGA": protein.data,
            "lgg_methTCGA": meth.data}
    
    # In[3]: Survival data
    #path = "C:/Users/da.salazarb/Google Drive/Tutorial_2018-10/03_Resultados/DataTCGA/DeepTCGA/"
    tcga["lgg_survivalTCGA"] = pd.read_csv(path+"/survivalTCGA.csv")
    tcga["lgg_survivalTCGA"].index = tcga["lgg_survivalTCGA"].bcr_patient_barcode + "-01"
    tcga["lgg_survivalTCGA"] = tcga["lgg_survivalTCGA"][['OS.time', 'DSS.time', 'DFI.time', 'PFI.time']]
    
    uniqueSamples = list(set.intersection(set(tcga["lgg_survivalTCGA"].index), set(cnv.data.index), set(mrna.data.index), list(protein.data.index), list(meth.data.index)))
    
    for i in range(0,len(tcga.keys())):
        tcga[(list(tcga.keys())[i])] = tcga[(list(tcga.keys())[i])].loc[uniqueSamples]
    
    return tcga