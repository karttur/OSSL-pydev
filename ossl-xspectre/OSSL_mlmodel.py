#!/Applications/anaconda3/envs/spectraimagine_py38/bin/python3.8
'''
Created on 3 Aug 2023

Last edited 17 Spet 2023

@author: thomasgumbricht

Notes
-----
The module plot.py:

    requires that you have soil spectra data organised as json files in xSpectre format.

    The script takes 3 string parameters as input:

        - docpath: the full path to a folder that must contain the txt file as given by the "projFN" parameter
        - projFN: the name of an existing txt files that sequentially lists json parameter files to run
        - jsonpath: the relative path (vis-a-vis "docpath") where the json parameter files (listed in "projFN") are

    The parameter files must list approximately 40 parameters in a precise nested json structure with dictionaries and lists.
    You can create a template json parameter file by running "def CreateParamJson" (just uncomment under "def SetupProcesses",
    this creates a template json parameter file called "extract_soillines.json" in the path given as the parameter "docpath".

    With an edited json parameter file the script reads the spectral data in xSpectre´s json format.
    The script first run the stand alone "def SetupProcesses" that reads the txt file "projFN" and
    then sequentialy run the json parameter files listed.

    Each soilline extraction (i.e. each json parameter file) is run as a separate instance of the class "SoilLine".

    Each soilline extract process result in 2 json files, containg 1) the extacted soillines and 2) soil
    spectral endmembers for darksoil and lightsoil. The names of the destination files cannot be set by the
    user, they are defaulted as follows:

    soillines result files:

        "rootFP"#visnirjson#visnir_OSSL_"region"_"date"_"first wavelength"-"last wavelength"_"band width"_result-soillines.json

    endmember result files:

        "rootFP"#visnirjson#visnir_OSSL_"region"_"date"_"first wavelength"-"last wavelength"_"band width"_result-endmembers.json

    If requested the script also produced png images showing the raw and/or final soillines:

        "rootFP"#visnirjson#visnir_OSSL_"region"_"date"_"first wavelength"-"last wavelength"_"band width"_raw-soillines.png
        "rootFP"#visnirjson#visnir_OSSL_"region"_"date"_"first wavelength"-"last wavelength"_"band width"_final-soillines.png

'''

# Standard library imports
import sys

import os

import json

import datetime

from copy import deepcopy

from math import sqrt

import pprint

import csv

# Third party imports
import tempfile

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import randint as sp_randint

from scipy.stats import boxcox

import pickle

from sklearn.preprocessing import scale, StandardScaler, QuantileTransformer, PowerTransformer, normalize
from sklearn.decomposition import PCA
  
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline, make_pipeline
from joblib import Memory

from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# Outlier detection
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Feature selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SelectFromModel

from sklearn.inspection import permutation_importance
from sklearn.metrics._regression import mean_absolute_error,\
    mean_absolute_percentage_error, median_absolute_error
from _ast import If

from cubist import Cubist

def Today():

    return datetime.datetime.now().date().strftime("%Y%m%d")

def ReadCSV(FPN):
    ''' Standard reader for all OSSL csv data files

    :param FPN: path to csv file
    :type FPN: str

    :return headers: list of columns
    :rtype: list

    :return rowL: array of data
    :rtype: list of list
    '''

    rowL = []

    with open( FPN, 'r' ) as csvF:

        reader = csv.reader(csvF)

        headers = next(reader, None)

        for row in reader:

            rowL.append(row)

    return headers, rowL

def DumpAnyJson(dumpD, jsonFPN):
    ''' dump, any json object

    :param exportD: formatted dictionary
    :type exportD: dict
    '''

    jsonF = open(jsonFPN, "w")

    json.dump(dumpD, jsonF, indent = 2)

    jsonF.close()

def ReadAnyJson(jsonFPN):
    """ Read json parameter file

    :param jsonFPN: path to json file
    :type jsonFPN: str

    :return paramD: parameters
    :rtype: dict
   """

    with open(jsonFPN) as jsonF:

        jsonD = json.load(jsonF)

    return (jsonD)

def CampaignParams():
    """ Default campaign parameters for all OSSL processing

        :returns: parameter dictionary

        :rtype: dict
    """

    campaignD = {'campaignId': 'OSSL-region-etc','campaignShortId':'OSSL-xyz'}

    campaignD['campaignType'] = 'laboratory'

    campaignD['theme'] = 'soil'

    campaignD['product'] = 'diffuse reflectance'

    campaignD['units'] = 'fraction'

    campaignD['geoRegion'] = "Sweden"

    return campaignD

def StandardParams():
    """ Default standard parameters for all OSSL processing

        :returns: parameter dictionary

        :rtype: dict
    """

    paramD = {}

    paramD['verbose'] = 1

    paramD['id'] = "auto"

    paramD['name'] = "auto"

    paramD['userId'] = "youruserid - any for now"

    paramD['importVersion'] = "OSSL-202308"

    return paramD

def StandardXspectreParams():
    """ Default standard parameters for all OSSL processing

        :returns: parameter dictionary

        :rtype: dict
    """

    paramD = {}

    paramD['verbose'] = 1

    paramD['id'] = "auto"

    paramD['name'] = "auto"

    paramD['userId'] = "youruserid - any for now"

    paramD['importVersion'] = "importxspectrev080"

    return paramD

def MLmodelParams():
    ''' Default parameters for soilline extraction from soil spectral library data

        :returns: parameter dictionary
        :rtype: dict
    '''

    paramD = StandardParams()

    paramD['campaign'] = CampaignParams()

    paramD['input'] = {}

    paramD['input']['jsonSpectraDataFilePath'] = 'path/to/jsonfile/with/spectraldata.json'

    paramD['input']['jsonSpectraParamsFilePath'] = 'path/to/jsonfile/with/spectralparams.json'

    paramD['input']['hyperParameterRandomTuning'] = 'path/to/jsonfile/with/hyperparam/tuning.json'

    ''' LUCAS oriented targetFeatures data'''
    paramD['targetFeatures'] = ['caco3_usda.a54_w.pct',
      'cec_usda.a723_cmolc.kg',
      'cf_usda.c236_w.pct',
      'clay.tot_usda.a334_w.pct',
      'ec_usda.a364_ds.m',
      'k.ext_usda.a725_cmolc.kg',
      'n.tot_usda.a623_w.pct',
      'oc_usda.c729_w.pct',
      'p.ext_usda.a274_mg.kg',
      'ph.cacl2_usda.a481_index',
      'ph.h2o_usda.a268_index',
      'sand.tot_usda.c60_w.pct',
      'silt.tot_usda.c62_w.pct']

    paramD['targetFeatureSymbols'] = {'caco3_usda.a54_w.pct':{'color': 'orange', 'size':50}}

    paramD['derivatives'] = {'apply':False, 'join':False}
    
    paramD['scattercorrection'] = {}
    
    paramD['scattercorrection']['comment'] = "scattercorr can apply the following: norm-1 (l1), norm-2 (l2) norm-max (max), snv (snv) or msc (msc). If two are gicen they will be done in series using output from the first as input to the second"

    paramD['scattercorrection']['apply'] = False
    
    paramD['scattercorrection']['scaler'] = []
    
    paramD['standardisation'] = {}
    
    paramD['standardisation']['comment'] = "remove mean spectral signal and optionallay scale using standard deviation for each band"

    paramD['standardisation']['apply'] = False
    
    paramD['standardisation']['meancentring'] = True
    
    paramD['standardisation']['unitscaling'] = False
    
    paramD['standardisation']['paretoscaling'] = False
    
    paramD['standardisation']['poissonscaling'] = False
    
    paramD['pcaPreproc'] = {}
    
    paramD['pcaPreproc']['comment'] = "remove mean spectral signal for each band"

    paramD['pcaPreproc']['apply'] = False
    
    paramD['pcaPreproc']['n_components'] = 0


    paramD['removeOutliers'] = {}

    paramD['removeOutliers']['comment'] = "removes sample outliers based on spectra only - globally applied as preprocess"

    paramD['removeOutliers']['apply'] = True

    paramD['removeOutliers']['detectorMethodList'] = ["iforest (isolationforest)",
                                                      "ee (eenvelope,ellipticenvelope)",
                                                      "lof (lofactor,localoutlierfactor)",
                                                      "1csvm (1c-svm, oneclasssvm)"]

    paramD['removeOutliers']['detector'] = "1csvm"

    paramD['removeOutliers']['contamination'] = 0.1

    paramD['manualFeatureSelection'] = {}

    paramD['manualFeatureSelection']['comment'] = "Manual feature selection overrides other selection alternatives"

    paramD['manualFeatureSelection'] ['apply'] = False

    paramD['manualFeatureSelection']['spectra'] = [ "A", "B", "C"],

    paramD['manualFeatureSelection']['derivatives'] = {}

    paramD['manualFeatureSelection']['derivatives']['firstWaveLength'] = ['A','D']

    paramD['manualFeatureSelection']['derivatives']['lastWaveLength'] = ['C','F']

    paramD['globalFeatureSelection'] = {}

    paramD['globalFeatureSelection']['comment'] ="removes spectra with variance below given thresholds - globally applied as preprocess",

    paramD['globalFeatureSelection']['apply'] = False

    paramD['globalFeatureSelection']['varianceThreshold'] = {'threshold': 0.025}

    paramD['modelFeatureSelection'] = {}

    paramD['modelFeatureSelection']['comment'] = 'feature selection using model data',

    paramD['modelFeatureSelection']['apply'] = False

    paramD['modelFeatureSelection']['univariateSelection'] = {}

    paramD['modelFeatureSelection']['univariateSelection']['apply'] = False

    paramD['modelFeatureSelection']['univariateSelection']['SelectKBest'] = {}

    paramD['modelFeatureSelection']['univariateSelection']['SelectKBest']['apply'] = False

    paramD['modelFeatureSelection']['univariateSelection']['SelectKBest']['n_features'] = 5


    paramD['modelFeatureSelection']['univariateSelection']['SelectPercentile'] = {}

    paramD['modelFeatureSelection']['univariateSelection']['SelectPercentile']['implemented'] = False

    paramD['modelFeatureSelection']['univariateSelection']['SelectPercentile']['apply'] = False

    paramD['modelFeatureSelection']['univariateSelection']['SelectPercentile']['percentile'] = 10


    paramD['modelFeatureSelection']['univariateSelection']['genericUnivariateSelect'] = {}

    paramD['modelFeatureSelection']['univariateSelection']['genericUnivariateSelect']['implemented'] = False

    paramD['modelFeatureSelection']['univariateSelection']['genericUnivariateSelect']['apply'] = False

    paramD['modelFeatureSelection']['univariateSelection']['genericUnivariateSelect']['hyperParameters'] = {}


    paramD['modelFeatureSelection']['RFE'] = {}

    paramD['modelFeatureSelection']['RFE']['apply'] = True

    paramD['modelFeatureSelection']['RFE']['CV'] = True

    paramD['modelFeatureSelection']['RFE']['n_features_to_select'] = 5

    paramD['modelFeatureSelection']['RFE']['step'] = 1


    paramD['featureAgglomeration'] = {}

    paramD['featureAgglomeration']['apply'] = False

    paramD['featureAgglomeration']['agglomerativeClustering'] = {}

    paramD['featureAgglomeration']['agglomerativeClustering']['apply'] = False

    paramD['featureAgglomeration']['agglomerativeClustering']['implemented'] = False

    paramD['featureAgglomeration']['wardClustering'] = {}

    paramD['featureAgglomeration']['wardClustering']['apply'] = False

    paramD['featureAgglomeration']['wardClustering']['n_cluster'] = 0

    paramD['featureAgglomeration']['wardClustering']['affinity'] = 'euclidean'

    paramD['featureAgglomeration']['wardClustering']['tuneWardClustering'] = {}

    paramD['featureAgglomeration']['wardClustering']['tuneWardClustering']['apply'] = False

    paramD['featureAgglomeration']['wardClustering']['tuneWardClustering']['kfolds'] = 3

    paramD['featureAgglomeration']['wardClustering']['tuneWardClustering']['clusters'] = [2,
            3,4,5,6,7,8,9,10,11,12]

    paramD['hyperParameterTuning'] = {}

    paramD['hyperParameterTuning']['apply'] = False

    paramD['hyperParameterTuning']['fraction'] = 0.5

    paramD['hyperParameterTuning']['nIterSearch'] = 6

    paramD['hyperParameterTuning']['n_top'] = 3

    paramD['hyperParameterTuning']['randomTuning'] = {}

    paramD['hyperParameterTuning']['randomTuning']['apply'] = False

    paramD['hyperParameterTuning']['exhaustiveTuning'] = {}

    paramD['hyperParameterTuning']['exhaustiveTuning']['apply'] = False

    paramD['featureImportance'] = {}

    paramD['featureImportance']['apply'] = True

    paramD['featureImportance']['reportMaxFeatures'] = 12

    paramD['featureImportance']['permutationRepeats'] = 10

    paramD['modelling'] = {}

    paramD['modelling']['apply'] = True

    paramD['regressionModels'] = {}

    paramD['regressionModels']['OLS'] = {}

    paramD['regressionModels']['OLS']['apply'] = False

    paramD['regressionModels']['OLS']['hyperParams'] = {}

    paramD['regressionModels']['OLS']['hyperParams']['fit_intercept'] = False

    paramD['regressionModels']['TheilSen'] = {}

    paramD['regressionModels']['TheilSen']['apply'] = False

    paramD['regressionModels']['TheilSen']['hyperParams'] = {}

    paramD['regressionModels']['Huber'] = {}

    paramD['regressionModels']['Huber']['apply'] = False

    paramD['regressionModels']['Huber']['hyperParams'] = {}

    paramD['regressionModels']['KnnRegr'] = {}

    paramD['regressionModels']['KnnRegr']['apply'] = False

    paramD['regressionModels']['KnnRegr']['hyperParams'] = {}

    paramD['regressionModels']['DecTreeRegr'] = {}

    paramD['regressionModels']['DecTreeRegr']['apply'] = False

    paramD['regressionModels']['DecTreeRegr']['hyperParams'] = {}

    paramD['regressionModels']['SVR'] = {}

    paramD['regressionModels']['SVR']['apply'] = False

    paramD['regressionModels']['SVR']['hyperParams'] = {}

    paramD['regressionModels']['SVR']['hyperParams']['kernel'] = 'linear'

    paramD['regressionModels']['SVR']['hyperParams']['C'] = 1.5

    paramD['regressionModels']['SVR']['hyperParams']['epsilon'] = 0.05

    paramD['regressionModels']['RandForRegr'] = {}

    paramD['regressionModels']['RandForRegr']['apply'] = False

    paramD['regressionModels']['RandForRegr']['hyperParams'] = {}

    paramD['regressionModels']['RandForRegr']['hyperParams']['n_estimators'] = 30


    paramD['regressionModels']['MLP'] = {}

    paramD['regressionModels']['MLP']['apply'] = False

    paramD['regressionModels']['MLP']['hyperParams'] = {}

    paramD['regressionModels']['MLP']['hyperParams']['hidden_layer_sizes'] = [100,100]

    paramD['regressionModels']['MLP']['hyperParams']['max_iter'] = 200

    paramD['regressionModels']['MLP']['hyperParams']['tol'] = 0.001

    paramD['regressionModels']['MLP']['hyperParams']['epsilon'] = 1e-8

    paramD['modelTests'] = {}

    paramD['modelTests']['trainTest'] = {}

    paramD['modelTests']['trainTest']['apply'] = False

    paramD['modelTests']['trainTest']['testSize'] = 0.3

    paramD['modelTests']['trainTest']['plot'] = True

    paramD['modelTests']['trainTest']['marker'] = 's'


    paramD['modelTests']['Kfold'] = {}

    paramD['modelTests']['Kfold']['apply'] = False

    paramD['modelTests']['Kfold']['folds'] = 10

    paramD['modelTests']['Kfold']['plot'] = True

    paramD['modelTests']['Kfold']['marker'] = '.'


    paramD['plot'] = {}

    paramD['plot']['apply'] = True

    paramD['plot']['subPlots'] = {}

    paramD['plot']['subPlots']['singles'] = {}

    paramD['plot']['subPlots']['singles']['apply'] = True

    paramD['plot']['subPlots']['singles']['regressor'] = False

    paramD['plot']['subPlots']['singles']['targetFeature'] = False

    paramD['plot']['subPlots']['singles']['hyperParameters'] = False

    paramD['plot']['subPlots']['singles']['modelTests'] = False

    paramD['plot']['subPlots']['rows'] = {}

    paramD['plot']['subPlots']['rows']['apply'] = True

    paramD['plot']['subPlots']['rows']['regressor'] = False

    paramD['plot']['subPlots']['rows']['targetFeature'] = False

    paramD['plot']['subPlots']['rows']['hyperParameters'] = False

    paramD['plot']['subPlots']['rows']['modelTests'] = False

    paramD['plot']['subPlots']['columns'] = {}

    paramD['plot']['subPlots']['columns']['apply'] = True

    paramD['plot']['subPlots']['columns']['regressor'] = False

    paramD['plot']['subPlots']['columns']['targetFeature'] = False

    paramD['plot']['subPlots']['columns']['hyperParameters'] = False

    paramD['plot']['subPlots']['columns']['modelTests'] = False

    paramD['plot']['subPlots']['doubles'] = {}

    paramD['plot']['subPlots']['doubles']['apply'] = True

    paramD['plot']['subPlots']['doubles']['columns'] = "regressor, targetFeature, hyperParameters or modelTest"

    paramD['plot']['subPlots']['doubles']['rows'] = "regressor, targetFeature, hyperParameters or modelTest"




    paramD['plot']['figSize'] = {'x':0,'y':0}

    paramD['plot']['legend'] = False

    paramD['plot']['tightLayout'] = False

    paramD['plot']['scatter'] = {'size':50}



    paramD['plot']['text'] = {'x':0.6,'y':0.2}

    paramD['plot']['text']['bandWidth'] = True

    paramD['plot']['text']['samples'] = True

    paramD['plot']['text']['text'] = ''

    paramD['figure'] = {}

    paramD['figure']['apply'] = True


    return (paramD)

def CreateArrangeParamJson(jsonFPN, projFN, processstep):
    """ Create the default json parameters file structure, only to create template if lacking

        :param str dstrootFP: directory path

        :param str jsonpath: subfolder under directory path
    """

    def ExitMsgMsg(flag):

        if flag:

            exitstr = 'json parameter file already exists: %s\n' %(jsonFPN)

        else:

            exitstr = 'json parameter file created: %s\n' %(jsonFPN)

        exitstr += ' Edit the json file for your project and rename it to reflect the commands.\n'

        exitstr += ' Add the path of the edited file to your project file (%s).\n' %(projFN)

        exitstr += ' Then set createjsonparams to False in the main section and rerun script.'

        exit(exitstr)

    if processstep.lower() in ['model','mlmodel']:

        # Get the default import params
        paramD = MLmodelParams()

        # Set the json FPN
        jsonFPN = os.path.join(jsonFPN, 'template_model_ossl-spectra.json')

    if processstep.lower() in ['importxspectre','arrangexspectre']:

        pass
        '''
        # Get the default import params
        paramD = ImportXspectreParams()

        # Set the json FPN
        jsonFPN = os.path.join(jsonFP, 'template_import_xspectre-spectra.json')

        # Set the json FPN
        jsonFPN = os.path.join(jsonFP, 'template_model_ossl-spectra.json')
        '''

    if os.path.exists(jsonFPN):

        ExitMsgMsg(True)

    DumpAnyJson(paramD,jsonFPN)

    ExitMsgMsg(False)

def CheckMakeDocPaths(rootpath,arrangeddatafolder, jsonpath, sourcedatafolder=False):
    """ Create the default json parameters file structure, only to create template if lacking

        :param str dstrootFP: directory path

        :param str jsonpath: subfolder under directory path
    """

    if not os.path.exists(rootpath):

        exitstr = "The rootpath does not exists: %s" %(rootpath)

        exit(exitstr)

    if sourcedatafolder:

        srcFP = os.path.join(os.path.dirname(__file__),rootpath,sourcedatafolder)

        if not os.path.exists(srcFP):

            exitstr = "The source data path to the original OSSL data does not exists:\n %s" %(srcFP)

            exit(exitstr)

    dstRootFP = os.path.join(os.path.dirname(__file__),rootpath,arrangeddatafolder)

    if not os.path.exists(dstRootFP):

        os.makedirs(dstRootFP)

    jsonFP = os.path.join(dstRootFP,jsonpath)

    if not os.path.exists(jsonFP):

        os.makedirs(jsonFP)
        
    return dstRootFP, jsonFP

def ReadImportParamsJson(jsonFPN):
    """ Read the parameters for importing OSSL data

    :param jsonFPN: path to json file
    :type jsonFPN: str

    :return paramD: parameters
    :rtype: dict
   """

    return ReadAnyJson(jsonFPN)

def ReadProjectFile(dstRootFP,projFN):

    projFPN = os.path.join(dstRootFP,projFN)

    if not os.path.exists(projFPN):

        exitstr = 'EXITING, project file missing: %s.' %(projFPN)

        exit( exitstr )

    infostr = 'Processing %s' %(projFPN)

    print (infostr)

    '''
    # Open and read the text file linking to all json files defining the project
    with open(projFPN) as f:

        jsonL = f.readlines()
    
    # Clean the list of json objects from comments and whithespace etc
    jsonProcessObjectL = [os.path.join(jsonFP,x.strip())  for x in jsonL if len(x) > 10 and x[0] != '#']
    '''
    
    jsonProcessObjectD = ReadAnyJson(projFPN)

    return jsonProcessObjectD


def snv(input_data):
    ''' Perform Multiplicative scatter correction
    copied 20240311: https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/
    '''
  
    # Define a new array and populate it with the corrected data  
    output_data = np.zeros_like(input_data)
    
    for i in range(input_data.shape[0]):
 
        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
 
    return output_data

def msc(input_data, reference=None):
    ''' Perform Multiplicative scatter correction
    copied 20240311: https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/
    '''
 
    # mean centre correction
    
    for i in range(input_data.shape[0]):
        
        input_data[i,:] -= input_data[i,:].mean()
 
    # Get the reference spectrum. If not given, estimate it from the mean    
    if reference is None:    
        # Calculate mean
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference
 
    # Define a new array and populate it with the corrected data    
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data[i,:], 1, full=True)
        # Apply correction
        data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0] 
 
    return (data_msc, ref)

def ScatterCorrectDF(dataFrame, scattercorrection, columns):
    """ Scatter correction for spectral signals

        :returns: organised spectral derivates
        :rtype: pandas dataframe
    """
    
    normD = {}
    
    scatcorrDFD = {}
    
    normD['l1'] = { 'norm':'l1', 'label':'L1 norm'}
    normD['l2'] = {'norm':'l2', 'label':'L2 norm'}
    normD['max'] = {'norm':'max', 'label':'max norm'}
    normD['snv'] = {'norm':'max', 'label':'SNV'}
    normD['msc'] = {'norm':'max', 'label':'MSC'}
    
    meanSpectra = None
    
    for scatcorr in scattercorrection.singles:
        
        scatcorrDFD[scatcorr] = {'label': normD[scatcorr]['label']}
        
        if scatcorr in ['l1','l2','max']:
            
            X1 = normalize(dataFrame, norm=scatcorr) 
        
            scatcorrDFD[scatcorr]['DF'] = pd.DataFrame(data=X1, columns=columns)
            
        elif scatcorr == 'snv':
            
            X = np.array(dataFrame[columns])
            
            snvSpectra = snv(X)
            
            scatcorrDFD[scatcorr]['DF'] = pd.DataFrame(data=snvSpectra, columns=columns)
        
        elif scatcorr == 'msc':
            
            #X = dataFrame[columns]
            X = np.array(dataFrame[columns])
            
            mscSpectra, meanSpectra = msc(X)
            
            scatcorrDFD[scatcorr]['DF'] = pd.DataFrame(data=mscSpectra, columns=columns)
            
    for d in scattercorrection.duals:
        
        scatcorr = '%s+%s' %(d[0],d[1])
        
        label = '%s + %s' %(normD[d[0]]['label'], normD[d[1]]['label'])
        
        scatcorrDFD[scatcorr] = {'label': label}
        
        dualCorrDF = deepcopy(dataFrame)
        
        for i in range(2):
            
            corrector = d[i] 
             
            if corrector in ['l1','l2','max']:
                
                X = np.array(dualCorrDF[columns])
                
                X1 = normalize(X, norm=corrector) 
                
                dualCorrDF = pd.DataFrame(data=X1, columns=columns)
              
            elif corrector == 'snv':
                
                X = np.array(dualCorrDF[columns])
                
                snvSpectra = snv(X)
                
                dualCorrDF = pd.DataFrame(data=snvSpectra, columns=columns)
            
            elif scatcorr == 'msc':
                
                #X = dataFrame[columns]
                X = np.array(dualCorrDF[columns])
                
                mscSpectra, meanSpectra = msc(X)
                
                dualCorrDF = pd.DataFrame(data=mscSpectra, columns=columns)
                
        scatcorrDFD[scatcorr]['DF'] = dualCorrDF
            
    return scatcorrDFD, meanSpectra
                      
def SpectraScatterCorrectionFromDf(dataFrame,scattercorrection):
    """ Scatter correction for spectral signals

        :returns: organised spectral derivates
        :rtype: pandas dataframe
    """
    
    columns = [item for item in dataFrame]
    
    dataFrameD, scatCorrMeanSpectra = ScatterCorrectDF(dataFrame, scattercorrection, columns)
    
    key = next(iter(dataFrameD))

    return dataFrameD[key]['DF'], key, scatCorrMeanSpectra

def SetMultiCompDstFPNs(rootPath, arrangeDataPath, multiProjectComparisonD,targetFeatureSymbolsD):
    '''
    '''
    multiCompFP = os.path.join(rootPath,arrangeDataPath,'multicomp')
    
    if not os.path.exists(multiCompFP):
        
        os.makedirs(multiCompFP)
        
    multiCompProjectFP = os.path.join(multiCompFP, multiProjectComparisonD['prefix'])
    
    if not os.path.exists(multiCompProjectFP):
        
        os.makedirs(multiCompProjectFP)
        
    multiCompProjectImageFP = os.path.join(multiCompProjectFP, 'images')
    
    if not os.path.exists(multiCompProjectImageFP):
        
        os.makedirs(multiCompProjectImageFP)
        
    multiCompProjectJsonFP = os.path.join(multiCompProjectFP, 'json')
    
    if not os.path.exists(multiCompProjectJsonFP):
        
        os.makedirs(multiCompProjectJsonFP)
                   
    indexL = ['coefficientImportance','permutationImportance','trainTest','Kfold']
    
    multCompImagesFPND = {}
    
    multCompJsonSummaryFPND = {}
    
    for targetFeature in multiProjectComparisonD['targetFeatures']:
            
        #print ('targetFeature', targetFeature)
        
        multCompSummaryFN = '%s_%s.json' %(multiProjectComparisonD['prefix'],targetFeature)
        
        multCompJsonSummaryFPND[targetFeature] = os.path.join(multiCompProjectJsonFP, multCompSummaryFN)
                
        multCompImagesFPND[targetFeature] = {}
        
           
        for i in indexL:
           
            #print ('i',i)
                      
            multCompImagesFN = '%s_%s_%s.png' %(multiProjectComparisonD['prefix'],targetFeature, i)
            
            multCompImagesFPND[targetFeature][i] = os.path.join(multiCompProjectImageFP, multCompImagesFN)
              
    return multCompImagesFPND, multCompJsonSummaryFPND
    
def SetMultCompPlots(multiProjectComparisonD,targetFeatureSymbolsD, figCols):
    '''
    '''
    
    if figCols == 0:
        
        exit('Multi comparisson requres at least one feature importance or one model test')

    multCompPlotIndexL = []
    
    multCompPlotsColumnD = {}
    
    multCompFig = {}
    
    multCompAxs = {}
    
    regressionModelL = []
    
    # Set the regression models to include:
    
    for r,row in enumerate(multiProjectComparisonD['regressionModels']):

        if multiProjectComparisonD['regressionModels'][row]['apply']:
            
            regressionModelL.append(row)
            
    figRows = len(regressionModelL)
        
    # Set the columns to include
    if multiProjectComparisonD['featureImportance']['apply']:
        
        if multiProjectComparisonD['featureImportance']['permutationImportance']['apply']:
        
            multCompPlotsColumnD['permutationImportance'] = len(multCompPlotIndexL)
            multCompPlotIndexL.append('permutationImportance')
              
        if multiProjectComparisonD['featureImportance']['coefficientImportance']['apply']:
        
            multCompPlotsColumnD['coefficientImportance'] = len(multCompPlotIndexL)
            multCompPlotIndexL.append('coefficientImportance')
            
    if multiProjectComparisonD['modelTests']['apply']:
        
        if multiProjectComparisonD['modelTests']['trainTest']['apply']:
        
            multCompPlotsColumnD['trainTest'] = len(multCompPlotIndexL)
            multCompPlotIndexL.append('trainTest')
            
        if multiProjectComparisonD['modelTests']['Kfold']['apply']:
        
            multCompPlotsColumnD['Kfold'] = len(multCompPlotIndexL)
            multCompPlotIndexL.append('Kfold')
                       
    # Set the figure size
    if multiProjectComparisonD['plot']['figSize']['x'] == 0:
        
        xadd = multiProjectComparisonD['plot']['figSize']['xadd']

        figSizeX = 3 * figCols + xadd

    else:

        figSizeX =multiProjectComparisonD['plot']['figSize']['x']

    if multiProjectComparisonD['plot']['figSize']['y'] == 0:
        
        yadd = multiProjectComparisonD['plot']['figSize']['yadd']

        figSizeY = 3 * figRows + yadd

    else:

        figSizeY =multiProjectComparisonD['plot']['figSize']['y']
                
    # Create column plots for each trial, with rows showing different regressors
    for targetFeature in multiProjectComparisonD['targetFeatures']:
        
        #print ('    targetFeature',targetFeature)
        
        multCompFig[targetFeature] = {}; multCompAxs[targetFeature] = {}
        
        for index in multCompPlotIndexL:
            
            #print ('        index',index)

            multCompFig[targetFeature][index], multCompAxs[targetFeature][index] = plt.subplots(figRows, figCols, figsize=(figSizeX, figSizeY))

            if multiProjectComparisonD['plot']['tightLayout']:
    
                multCompFig[targetFeature][index].tight_layout()

            # Set subplot wspace and hspace
            if multiProjectComparisonD['plot']['hwspace']['wspace']:
    
                multCompFig[targetFeature][index].subplots_adjust(wspace=multiProjectComparisonD['plot']['hwspace']['wspace'])
    
            if multiProjectComparisonD['plot']['hwspace']['hspace']:
    
                multCompFig[targetFeature][index].subplots_adjust(hspace=multiProjectComparisonD['plot']['hwspace']['hspace'])
    
            label = targetFeatureSymbolsD['targetFeatureSymbols'][targetFeature]['label']
            
            if index in ['trainTest','Kfold']:
                
                suptitle = "Model test: %s; Target: %s (rows=regressors)\n" %(index, label)
            
            else:
                
                indextitle = '%s' %( index.replace('Importance', ' Importance'))
                
                suptitle = "Covariate evaluation: %s; Target: %s (rows=regressors)\n" %(indextitle, label)

            '''
            if self.varianceSelectTxt != None:
    
                suptitle += ', %s' %(self.varianceSelectTxt)
    
            if self.outlierTxt != None:
    
                suptitle +=  ', %s' %(self.outlierTxt)
            '''
            
            # Set suptitle
            multCompFig[targetFeature][index].suptitle( suptitle )
    
            # Set subplot titles, only for top row:
            # for r,row in enumerate(regressionModelL):
            #for r,row in enumerate([0]):
    
            #for c,col in enumerate(multCompPlotIndexL):
            for c in range(figCols): 
                
                modelNrStr = '%s' %(c)
            
                if modelNrStr in multiProjectComparisonD['trialid']:
                    
                    trialId = multiProjectComparisonD['trialid'][modelNrStr]
                
                else:
                
                    trialId = 'trial_%s' %(c)
                                
                if figRows == 1:
                
                    multCompAxs[targetFeature][index][c].set_title( trialId )
                
                else:
                    
                    multCompAxs[targetFeature][index][0][c].set_title( trialId )
                                          
    return (multCompFig, multCompAxs, multCompPlotsColumnD)



class Obj(object):
    ''' Convert json parameters to class objects
    '''

    def __init__(self, paramD):
        ''' Convert input parameters from nested dict to nested class object

            :param dict paramD: parameters
        '''
        for k, v in paramD.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [Obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Obj(v) if isinstance(v, dict) else v)

    def _SetArrangeDefautls(self):
        ''' Set class object default data if missing
        '''

        if not hasattr(self, 'sitedata'):

            setattr(self, 'sitedata', [])

        sitedataMinL = ["id.layer_local_c",
                        "dataset.code_ascii_txt",
                        "longitude.point_wgs84_dd",
                        "latitude.point_wgs84_dd",
                        "location.point.error_any_m",
                        "layer.upper.depth_usda_cm",
                        "layer.lower.depth_usda_cm",
                        "id_vis","id_mir"]

        for item in sitedataMinL:

            if not item in self.sitedata:

                self.sitedata.append(item)
        '''
        self.visnirStep = int(self.input.visnirOutputBandWidth/ self.input.visnirInputBandWidth)

        self.mirStep = int(self.input.mirOutputBandWidth/ self.input.mirInputBandWidth)

        self.neonStep = int(self.input.neonOutputBandWidth/ self.input.neonInputBandWidth)
        '''

    def _SetPlotDefaults(self):
        ''' Set class object default data if required
        '''

        if self.plot.singles.figSize.x == 0:

            self.plot.singles.figSize.x = 8

        if self.plot.singles.figSize.y == 0:

            self.plot.singles.figSize.y = 6

    def _SetPlotTextPos(self, plot, xmin, xmax, ymin, ymax):
        ''' Set position of text objects for matplotlib

            :param float xmin: x-axis minimum

            :param float xmax: x-axis maximum

            :param float ymin: y-axis minimum

            :param float ymax: y-axis maximum

            :returns: text x position
            :rtype: float

            :returns: text y position
            :rtype: float
        '''

        x = plot.text.x*(xmax-xmin)+xmin

        y = plot.text.y*(ymax-ymin)+ymin

        return (x,y)

    def _SetSoilLineDefautls(self):
        ''' Set class object default data if required
        '''

        if self.plot.singles.figSize.x == 0:

            self.plot.singles.figSize.x = 8

        if self.plot.singles.figSize.y == 0:

            self.plot.singles.figSize.y = 6

    def _SetModelDefaults(self):
        ''' Set class object default data if required
        '''

        if self.plot.singles.figSize.x == 0:

            self.plot.singles.figSize.x = 4

        if self.plot.singles.figSize.y == 0:

            self.plot.singles.figSize.y = 4

        # Check if Manual feature selection is set
        if self.manualFeatureSelection.apply:

            # Turn off the derivates alteratnive (done as part of the manual selection if requested)
            self.derivatives.apply = False

            # Turn off all other feature selection/agglomeration options
            self.globalFeatureSelection.apply = False

            self.modelFeatureSelection.apply = False

            self.featureAgglomeration.apply = False


def ReadModelJson(jsonFPN):
    """ Read the parameters for modeling

    :param jsonFPN: path to json file
    :type jsonFPN: str

    :return paramD: parameters
    :rtype: dict
    """
    
    if not os.path.exists(jsonFPN):
       
        print (jsonFPN)

    with open(jsonFPN) as jsonF:

        paramD = json.load(jsonF)

    return (paramD)

class RegressionModels:

    '''Machinelearning using regression models
    '''
    def __init__(self):
        '''creates an empty instance of RegressionMode
        '''

        self.modelSelectD = {}

        self.modelRetaindD = {}

        self.modD = {}

        #Create a list to hold retained columns
        self.retainD = {}

        self.retainPrintD = {}

        self.tunedModD = {}

    def _ExtractDataFrame(self):
        ''' Extract the original dataframe to X (covariate) array and y (predict) column
        '''

        # Extract the target feature
        self.y = self.abundanceDf[self.targetFeature]

        # Append the target array to the self.spectraDF dataframe
        self.spectraDF['target'] = self.y

        # define the list of covariates to use
        #self.columnsX = [item for item in self.spectraDF.columns if item not in self.omitL]
        self.columnsX = [item for item in self.spectraDF.columns]

        # extract the covariate columns as X
        self.X = self.spectraDF[self.columnsX]

        # Drop the added target column from the dataframe
        self.spectraDF = self.spectraDF.drop('target', axis=1)

        # Remove all samples where the targetfeature is NaN
        self.X = self.X[~np.isnan(self.X).any(axis=1)]
        
        # Drop the added target column from self.X
        self.X = self.X.drop('target', axis=1)

        # Then also delete NaN from self.y
        self.y = self.y[~np.isnan(self.y)]
        
        # Remove all non-finite values
        
        self.X = self.X[np.isfinite(self.y)] 
        
        self.y = self.y[np.isfinite(self.y)] 

    def _SetTargetFeatureSymbol(self):
        '''
        '''

        self.featureSymbolColor = 'black'
        
        self.featureSymbolAlpha = 1.0

        self.featureSymbolMarker = '.'

        self.featureSymbolSize = 100

        if hasattr(self, 'targetFeatureSymbols'):

            if hasattr(self.targetFeatureSymbols, self.targetFeature):

                symbol = getattr(self.targetFeatureSymbols, self.targetFeature)

                if hasattr(symbol, 'color'):

                    self.featureSymbolColor = getattr(symbol, 'color')
                    
                if hasattr(symbol, 'alpha'):

                    self.featureSymbolAlpha = getattr(symbol, 'alpha')

                if hasattr(symbol, 'size'):

                    self.featureSymbolSize = getattr(symbol, 'size')

    def _PlotRegr(self, obs, pred, suptitle, title, txtstr,  txtstrHyperParams, regrModel, modeltest, multCompAxs):
        '''
        '''
        if self.plot.singles.apply:

            fig, ax = plt.subplots()
            ax.scatter(obs, pred, edgecolors=(0, 0, 0),  color=self.featureSymbolColor,
                       alpha=self.featureSymbolAlpha,
                       s=self.paramD['regressionModelSymbols'][self.regrModel[0]]['size'],
                       marker=self.paramD['regressionModelSymbols'][self.regrModel[0]]['marker'])
            ax.plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)
            ax.set_xlabel('Observations')
            ax.set_ylabel('Predictions')
            plt.suptitle(suptitle)
            plt.title(title)
            plt.text(obs.min(), (obs.max()-obs.min())*0.8, txtstr,  wrap=True)

            #plt.text(obs.max()-((obs.max()-obs.min())*0.3), (obs.min()+obs.max())*0.1, txtstrHyperParams,  wrap=True)

            if self.plot.singles.screenShow:

                plt.show()

            if self.plot.singles.savePng:

                fig.savefig(self.imageFPND[self.targetFeature][regrModel][modeltest])

            plt.close(fig=fig)

        if self.plot.rows.apply:

            if self.plot.rows.targetFeatures.apply:

                # modeltest is either trainTest of Kfold
                if modeltest in self.plot.rows.targetFeatures.columns:

                    if len(self.targetFeatures) == 1:

                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[modeltest] ].scatter(obs, pred, edgecolors=(0, 0, 0),  
                                color=self.featureSymbolColor, alpha=self.featureSymbolAlpha,
                               s=self.paramD['regressionModelSymbols'][self.regrModel[0]]['size'],
                               marker=self.paramD['regressionModelSymbols'][self.regrModel[0]]['marker'])

                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[modeltest] ].plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)

                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[modeltest]].text(.05, .95,
                                                        txtstr, ha='left', va='top',
                                                        transform=self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[modeltest]].transAxes)

                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[modeltest]].yaxis.set_label_position("right")


                    else:

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest] ].scatter(obs, pred, edgecolors=(0, 0, 0),  color=self.featureSymbolColor,
                               s=self.paramD['regressionModelSymbols'][self.regrModel[0]]['size'],
                               marker=self.paramD['regressionModelSymbols'][self.regrModel[0]]['marker'])

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest] ].plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest]].text(.05, .95,
                                                        txtstr, ha='left', va='top',
                                                        transform=self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest]].transAxes)

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest]].yaxis.set_label_position("right")

                    # if at last column
                    if self.targetFeaturePlotColumnD[modeltest] == len(self.plot.rows.regressionModels.columns)-1:

                        if len(self.targetFeatures) == 1:

                            self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[modeltest]].set_ylabel('Predictions')

                        else:

                            self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest]].set_ylabel('Predictions')

                    # if at last row
                    if self.targetN == self.nTargetFeatures-1:

                        if len(self.targetFeatures) == 1:

                            self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[modeltest]].set_xlabel('Observations')

                        else:

                            self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest]].set_xlabel('Observations')


            if self.plot.rows.regressionModels.apply:

                # modeltest is either trainTest of Kfold
                if modeltest in self.plot.rows.regressionModels.columns:

                    #self.columnAxs[self.regrModel][self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest] ].scatter(obs, pred, edgecolors=(0, 0, 0),  color=self.featureSymbolColor,
                    #       s=self.featureSymbolSize, marker=self.featureSymbolMarker)

                    if (len(self.regressorModels)) == 1:

                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest] ].scatter(obs, pred, edgecolors=(0, 0, 0),  
                               color=self.featureSymbolColor, alpha=self.featureSymbolAlpha,
                               s=self.paramD['regressionModelSymbols'][self.regrModel[0]]['size'],
                               marker=self.paramD['regressionModelSymbols'][self.regrModel[0]]['marker'])

                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)


                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].text(.05, .95, txtstr, ha='left', va='top',
                                                        transform=self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].transAxes)

                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].yaxis.set_label_position("right")

                    else:

                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest] ].scatter(obs, pred, edgecolors=(0, 0, 0),  
                                color=self.featureSymbolColor, alpha=self.featureSymbolAlpha,
                               s=self.paramD['regressionModelSymbols'][self.regrModel[0]]['size'],
                               marker=self.paramD['regressionModelSymbols'][self.regrModel[0]]['marker'])

                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)


                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].text(.05, .95, txtstr, ha='left', va='top',
                                                        transform=self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].transAxes)

                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].yaxis.set_label_position("right")


                    # if at last column
                    if self.regressionModelPlotColumnD[modeltest] == len(self.plot.rows.targetFeatures.columns)-1:

                        if self.regrN == self.nRegrModels-1:

                            if (len(self.regressorModels)) == 1:

                                self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].set_ylabel('Predictions')

                            else:

                                self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].set_ylabel('Predictions')


                        else:

                            if (len(self.regressorModels)) == 1:

                                self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].set_ylabel('Predictions')

                            else:

                                self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].set_ylabel('Predictions')


                    # if at last row
                    if self.regrN == self.nRegrModels-1:

                        if (len(self.regressorModels)) == 1:

                            self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].set_xlabel('Observations')

                        else:

                            self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].set_xlabel('Observations')

                    else:

                        if (len(self.regressorModels)) == 1:

                            self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].set_xlabel('Observations')

                        else:

                            self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].set_xlabel('Observations')
        
        if self.multcompplot:
 
            index = modeltest
            columnNr = self.modelNr
            
            if (len(self.regressorModels)) == 1: # only 1 row in subplot

                multCompAxs[self.targetFeature][index][columnNr].scatter(obs, pred, edgecolors=(0, 0, 0),  
                        color=self.featureSymbolColor, alpha=self.featureSymbolAlpha,
                       s=self.paramD['regressionModelSymbols'][self.regrModel[0]]['size'],
                       marker=self.paramD['regressionModelSymbols'][self.regrModel[0]]['marker'])

                multCompAxs[self.targetFeature][index][columnNr].plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)


                multCompAxs[self.targetFeature][index][columnNr].text(.05, .95, txtstr, ha='left', va='top',
                                                transform=multCompAxs[self.targetFeature][index][columnNr].transAxes)

                multCompAxs[self.targetFeature][index][columnNr].yaxis.set_label_position("right")

            else:

                multCompAxs[self.targetFeature][index][self.regrN, columnNr ].scatter(obs, pred, edgecolors=(0, 0, 0),  
                        color=self.featureSymbolColor, alpha=self.featureSymbolAlpha,
                       s=self.paramD['regressionModelSymbols'][self.regrModel[0]]['size'],
                       marker=self.paramD['regressionModelSymbols'][self.regrModel[0]]['marker'])

                multCompAxs[self.targetFeature][index][self.regrN, columnNr].plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)

                multCompAxs[self.targetFeature][index][self.regrN, columnNr].text(.05, .95, txtstr, ha='left', va='top',
                                                transform=multCompAxs[self.targetFeature][index][self.regrN, columnNr].transAxes)

                multCompAxs[self.targetFeature][index][self.regrN, columnNr].yaxis.set_label_position("right")

            # if at first column
            if columnNr == 0:

                if (len(self.regressorModels)) == 1:
                        
                    multCompAxs[self.targetFeature][index][columnNr].set_ylabel(self.regrModel[0])
                    multCompAxs[self.targetFeature][index][columnNr].yaxis.set_label_position("left")

                else:

                    multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_ylabel(self.regrModel[0])
                    multCompAxs[self.targetFeature][index][self.regrN, columnNr].yaxis.set_label_position("left")

            # if at last column
            if columnNr+1 == multCompAxs[self.targetFeature][index].shape[0]:


                if (len(self.regressorModels)) == 1:

                    multCompAxs[self.targetFeature][index][columnNr].set_ylabel('Predictions')

                else:

                    multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_ylabel('Predictions')



            # if at last row
            if self.regrN == self.nRegrModels-1:

                if (len(self.regressorModels)) == 1:

                    multCompAxs[self.targetFeature][index][columnNr].set_xlabel('Observations')

                else:

                    multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_xlabel('Observations')
                    
    def _RegModelSelectSet(self):
        """ Set the regressors to evaluate
        """

        self.regressorModels = []

        if hasattr(self.regressionModels, 'OLS') and self.regressionModels.OLS.apply:

            self.regressorModels.append(('OLS', linear_model.LinearRegression(**self.jsonparamsD['regressionModels']['OLS']['hyperParams'])))

            self.modelSelectD['OLS'] = []

        if hasattr(self.regressionModels, 'TheilSen') and self.regressionModels.TheilSen.apply:

            self.regressorModels.append(('TheilSen', linear_model.TheilSenRegressor(**self.jsonparamsD['regressionModels']['OLS']['hyperParams'])))

            self.modelSelectD['TheilSen'] = []

        if hasattr(self.regressionModels, 'Huber') and self.regressionModels.Huber.apply:

            self.regressorModels.append(('Huber', linear_model.HuberRegressor(**self.jsonparamsD['regressionModels']['OLS']['hyperParams'])))

            self.modelSelectD['Huber'] = []

        if hasattr(self.regressionModels, 'KnnRegr') and self.regressionModels.KnnRegr.apply:
            self.regressorModels.append(('KnnRegr', KNeighborsRegressor( **self.jsonparamsD['regressionModels']['KnnRegr']['hyperParams'])))
            self.modelSelectD['KnnRegr'] = []

        if hasattr(self.regressionModels, 'DecTreeRegr') and self.regressionModels.DecTreeRegr.apply:
            self.regressorModels.append(('DecTreeRegr', DecisionTreeRegressor(**self.jsonparamsD['regressionModels']['DecTreeRegr']['hyperParams'])))
            self.modelSelectD['DecTreeRegr'] = []

        if hasattr(self.regressionModels, 'SVR') and self.regressionModels.SVR.apply:
            self.regressorModels.append(('SVR', SVR(**self.jsonparamsD['regressionModels']['SVR']['hyperParams'])))
            self.modelSelectD['SVR'] = []

        if hasattr(self.regressionModels, 'RandForRegr') and self.regressionModels.RandForRegr.apply:
            self.regressorModels.append(('RandForRegr', RandomForestRegressor( **self.jsonparamsD['regressionModels']['RandForRegr']['hyperParams'])))
            self.modelSelectD['RandForRegr'] = []

        if hasattr(self.regressionModels, 'MLP') and self.regressionModels.MLP.apply:

            '''
            # First make a pipeline with standardscaler + MLP
            mlp = make_pipeline(
                StandardScaler(),
                MLPRegressor( **self.jsonparamsD['regressionModels']['MLP']['hyperParams'])
            )
            '''
            mlp = Pipeline([('scl', StandardScaler()),
                    ('clf', MLPRegressor( **self.jsonparamsD['regressionModels']['MLP']['hyperParams']) ) ])

            # Then add the pipeline as MLP
            self.regressorModels.append(('MLP', mlp))

            self.modelSelectD['MLP'] = []
        
        if hasattr(self.regressionModels, 'Cubist') and self.regressionModels.Cubist.apply:
            self.regressorModels.append(('Cubist', Cubist( **self.jsonparamsD['regressionModels']['Cubist']['hyperParams'])))
            self.modelSelectD['Cubist'] = []
        
    def _CheckParams(self, jsonProcessFN):
        ''' Check parameters
        '''
        
        if not hasattr(self,'targetFeatures'):
            exitStr = 'Exiting: the modelling process file %s\n    has not targetFeature' %(jsonProcessFN)
            exit(exitStr)
        
    def _RegrModTrainTest(self, multCompAxs):
        '''
        '''

        #Retrieve the model name and the model itself
        name,model = self.regrModel

        #Split the data into training and test subsets
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=self.modelTests.trainTest.testSize)

        #Fit the model
        model.fit(X_train, y_train)

        #Predict the independent variable in the test subset
        predict = model.predict(X_test)
        
        r2_total = r2_score(y_test, predict)
        
        rmse_total = sqrt(mean_squared_error(y_test, predict))
        
        medae_total = median_absolute_error(y_test, predict)
        
        mae_total = mean_absolute_error(y_test, predict)
        
        mape_total = mean_absolute_percentage_error(y_test, predict)
        
        

        self.trainTestResultD[self.targetFeature][name] = {'rmse':rmse_total,
                                                           'mae':mae_total,
                                                           'medae': medae_total,
                                                           'mape':mape_total,
                                                           'r2': r2_total,
                                                           'hyperParameterSetting': self.jsonparamsD['regressionModels'][name]['hyperParams'],
                                                           'pickle': self.trainTestPickleFPND[self.targetFeature][name]
                                                           }
        
        self.trainTestSummaryD[self.targetFeature][name] = {'rmse':rmse_total,
                                                           'mae':mae_total,
                                                           'medae': medae_total,
                                                           'mape':mape_total,
                                                           'r2': r2_total,
                                                           }
        
        # Set regressor scores to 3 decimals
        self.trainTestResultD[self.targetFeature][name] = {k:(round(v,3) if isinstance(v,float) else v) for (k,v) in self.trainTestResultD[self.targetFeature][name].items()}

        self.trainTestSummaryD[self.targetFeature][name] = {k:(round(v,3) if isinstance(v,float) else v) for (k,v) in self.trainTestSummaryD[self.targetFeature][name].items()}

        # Save the complete model with cPickle
        pickle.dump(model, open(self.trainTestPickleFPND[self.targetFeature][name],  'wb'))

        if self.verbose:

            infoStr =  '                trainTest Model: %s\n' %(name)
            infoStr += '                    hyperParams: %s\n' %(self.jsonparamsD['regressionModels'][name]['hyperParams'])

            infoStr += '                    Root mean squared error (RMSE) total: %.2f\n' % rmse_total
            infoStr += '                    Variance (r2) score total: %.2f\n' % r2_total
            
            infoStr += '                    Mean absolute error (MAE) total: %.2f\n' %( mae_total)

            infoStr += '                    Mean absolute percent error (MAPE) total: %.2f\n' %( mape_total)

            infoStr += '                    Median absolute error (MedAE) total: %.2f\n' %( medae_total)

            print (infoStr)

        if self.modelTests.trainTest.plot:
            txtstr = 'nspectra: %s\n' %(self.X.shape[0])
            txtstr += 'nbands: %s\n' %(self.X.shape[1])
            #txtstr += 'min wl: %s\n' %(self.bandL[0])
            #txtstr += 'max wl: %s\n' %(self.bandL[len(self.bandL)-1])
            #txtstr += 'bands: %s\n' %( ' ,'.join('({0})'.format(w) for w in self.aggBandL)  )
            #txtstr += 'width wl: %s' %(int(self.bandL[1]-self.bandL[0]))

            #txtstrHyperParams =  self.HPtuningtxt+'\nHyper Parameters:\n'+'\n'.join([key+': '+str(value) for key, value in self.tunedModD[name].items()])
            suptitle = '%s train/test model (testsize = %s)' %(self.targetFeature, self.modelTests.trainTest.testSize)
            title = ('Model: %(mod)s; RMSE: %(rmse)2f; r2: %(r2)2f' \
                      % {'mod':name,'rmse':mean_squared_error(y_test, predict),'r2': r2_score(y_test, predict)} )

            txtstr = ('RMSE: %(rmse)2f\nr2: %(r2)2f\n nSamples: %(n)d' \
                      % {'rmse':self.trainTestResultD[self.targetFeature][name]['rmse'],
                         'r2': self.trainTestResultD[self.targetFeature][name]['r2'],
                         'n': self.X.shape[0]} )

            self._PlotRegr(y_test, predict, suptitle, title, txtstr, '',name, 'trainTest', multCompAxs)

    def _RegrModKFold(self, multCompAxs):
        """
        """

        #Retrieve the model name and the model itself
        name,model = self.regrModel

        predict = model_selection.cross_val_predict(model, self.X, self.y, cv=self.modelTests.Kfold.folds)

        rmse_total = sqrt(mean_squared_error(self.y, predict))

        r2_total = r2_score(self.y, predict)
        
        scoring = 'r2'

        r2_folded = model_selection.cross_val_score(model, self.X, self.y, cv=6, scoring=scoring)
        
        scoring = 'neg_mean_absolute_error'
        
        mae_folded = model_selection.cross_val_score(model, self.X, self.y, cv=6, scoring=scoring)

        scoring = 'neg_mean_absolute_percentage_error'
        
        mape_folded = model_selection.cross_val_score(model, self.X, self.y, cv=6, scoring=scoring)

        scoring = 'neg_median_absolute_error'
        
        medae_folded = model_selection.cross_val_score(model, self.X, self.y, cv=6, scoring=scoring)

        scoring = 'neg_root_mean_squared_error'
        
        rmse_folded = model_selection.cross_val_score(model, self.X, self.y, cv=6, scoring=scoring)

        self.KfoldResultD[self.targetFeature][name] = {'rmse_total': rmse_total,
                                                       'r2_total': r2_total,
                                                       
                                                       'rmse_folded_mean': -1*rmse_folded.mean(),
                                                       'rmse_folded_std': rmse_folded.std(),
                                                       
                                                       'mae_folded_mean': -1*mae_folded.mean(),
                                                       'mae_folded_std': mae_folded.std(),
                                                       
                                                       'mape_folded_mean': -1*mape_folded.mean(),
                                                       'mape_folded_std': mape_folded.std(),
                                                       
                                                       'medae_folded_mean': medae_folded.mean(),
                                                       'medae_folded_std': medae_folded.std(),
                                                       
                                                        'r2_folded_mean': r2_folded.mean(),
                                                        'r2_folded_std': r2_folded.std(),
                                                        'hyperParameterSetting': self.jsonparamsD['regressionModels'][name]['hyperParams'],
                                                        'pickle': self.KfoldPickleFPND[self.targetFeature][name]
                                                        }
        
        self.KfoldSummaryD[self.targetFeature][name] = {'rmse_total': rmse_total,
                                                       'r2_total': r2_total,
                                                       
                                                       'rmse_folded_mean': -1*rmse_folded.mean(),
                                                       'rmse_fodled_std': rmse_folded.std(),
                                                       
                                                       'mae_folded_mean': -1*mae_folded.mean(),
                                                       'mae_folded_std': mae_folded.std(),
                                                       
                                                       'mape_folded_mean': -1*mape_folded.mean(),
                                                       'mape_folded_std': mape_folded.std(),
                                                       
                                                
                                                       'medae_folded_mean': medae_folded.mean(),
                                                       'medae_folded_std': medae_folded.std(),
                                                       
                                                 
                                                        'r2_folded_mean': r2_folded.mean(),
                                                        'r2_folded_std': r2_folded.std(),
                                                        }
        
        # Set regressor scores to 3 decimals
        self.KfoldResultD[self.targetFeature][name] = {k:(round(v,3) if isinstance(v,float) else v) for (k,v) in self.KfoldResultD[self.targetFeature][name].items()}

        self.KfoldSummaryD[self.targetFeature][name] = {k:(round(v,3) if isinstance(v,float) else v) for (k,v) in self.KfoldSummaryD[self.targetFeature][name].items()}

        
        # Save the complete model with cPickle
        pickle.dump(model, open(self.KfoldPickleFPND[self.targetFeature][name],  'wb'))

        if self.verbose:

            infoStr =  '                Kfold Model: %s\n' %(name)
            infoStr += '                    hyperParams: %s\n' %(self.jsonparamsD['regressionModels'][name]['hyperParams'])
            infoStr += '                    Root mean squared error (RMSE) total: %.2f\n' % rmse_total
            infoStr += '                    Variance (r2) score total: %.2f\n' % r2_total
            
            infoStr += '                    RMSE folded: %.2f (%.2f) \n' %( -1*rmse_folded.mean(),  rmse_folded.std())
            
            infoStr += '                    Mean absolute error (MAE) folded: %.2f (%.2f) \n' %( -1*mae_folded.mean(),  mae_folded.std())

            infoStr += '                    Mean absolute percent error (MAPE) folded: %.2f (%.2f) \n' %( -1*mape_folded.mean(),  mape_folded.std())

            infoStr += '                    Median absolute error (MedAE) folded: %.2f (%.2f) \n' %( -1*medae_folded.mean(),  medae_folded.std())

            infoStr += '                    Variance (r2) score folded: %.2f (%.2f) \n' %( r2_folded.mean(),  r2_folded.std())

            print (infoStr)


        txtstr = 'nspectra: %s\n' %(self.X.shape[0])
        txtstr += 'nbands: %s\n' %(self.X.shape[1])
        #txtstr += 'min wl: %s\n' %(self.bandL[0])
        #txtstr += 'max wl: %s\n' %(self.bandL[len(self.bandL)-1])
        #txtstr += 'bands: %s\n' %( ' ,'.join('({0})'.format(w) for w in self.aggBandL)  )
        #txtstr += 'width wl: %s' %(int(self.bandL[1]-self.bandL[0]))

        #txtstrHyperParams =  self.HPtuningtxt+'\nHyper Parameters:\n'+'\n'.join([key+': '+str(value) for key, value in self.tunedModD[name].items()])
        suptitle = '%s Kfold model (nfolds = %s)' %(self.targetFeature, self.modelTests.Kfold.folds)
        title = ('Model: %(mod)s; RMSE: %(rmse)2f; r2: %(r2)2f' \
                  % {'mod':name,'rmse':rmse_total,'r2': r2_total} )

        txtstr = ('RMSE: %(rmse)2f\nr2: %(r2)2f\nSamples: %(n)d' \
                      % {'rmse':self.KfoldResultD[self.targetFeature][name]['rmse_total'],
                         'r2': self.KfoldResultD[self.targetFeature][name]['r2_total'],
                         'n': self.X.shape[0]} )

        self._PlotRegr(self.y, predict, suptitle, title, txtstr, '',name, 'Kfold', multCompAxs)

    def _PlotFeatureImportanceSingles(self, featureArray, importanceArray, errorArray, title, xyLabel, pngFPN):
        '''
        '''
        # Convert to a pandas series
        importanceDF = pd.Series(importanceArray, index=featureArray)

        singlefig, ax = plt.subplots()

        if isinstance(errorArray, np.ndarray):

            importanceDF.plot.bar(yerr=errorArray, color=self.featureSymbolColor, ax=ax)

        else:
            importanceDF.plot.bar(yerr=errorArray, color=self.featureSymbolColor, ax=ax)

        ax.set_title(title)

        if xyLabel[0]:

            ax.set_ylabel(xyLabel[0])

        if xyLabel[1]:

            ax.set_ylabel(xyLabel[1])

        if self.plot.tightLayout:

            singlefig.tight_layout()

        if self.plot.singles.screenShow:

            plt.show()

        if self.plot.singles.savePng:

            #fig.savefig(self.imageFPND[self.targetFeature][name]['featureImportance']['permutationImportance'])
            singlefig.savefig(pngFPN)

        plt.close(fig=singlefig)

    def _PlotFeatureImportanceRows(self, featureArray, importanceArray, errorArray, importanceCategory, yLabel):
        '''
        '''

        nnFS = self.X.shape

        text = 'nFeatures: %s' %(nnFS[1])

        if self.targetFeatureSelectionTxt != None:

            text += '\n%s' %(self.targetFeatureSelectionTxt)

        if self.agglomerateTxt != None:

            text += '\n%s' %(self.agglomerateTxt)

        if self.modelFeatureSelectionTxt != None:

            text += '\n%s' %(self.modelFeatureSelectionTxt)

        if self.plot.rows.targetFeatures.apply:

            if importanceCategory in self.plot.rows.targetFeatures.columns:

                if (len(self.targetFeatures)) == 1:

                    self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory] ].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

                    self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory] ].tick_params(labelleft=False)

                    self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].text(.3, .95, text, ha='left', va='top',
                                                    transform=self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].transAxes)

                    self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].set_ylabel(yLabel)

                else:

                    self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory] ].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

                    self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory] ].tick_params(labelleft=False)

                    self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].text(.3, .95, text, ha='left', va='top',
                                                    transform=self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].transAxes)

                    self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].set_ylabel(yLabel)


                if importanceCategory == 'featureImportance':

                    if (len(self.targetFeatures)) == 1:

                        # Draw horisontal line ay y=y
                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].axhline(y=0, lw=1, c='black')

                    else:

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].axhline(y=0, lw=1, c='black')
                #x,y = SetTextPos(self.plot.text.x, self.plot.text.y, self.xylimD['xmin'], self.xylimD['xmax'], self.xylimD['ymin'], self.xylimD['ymax'])

                #self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD['permutationImportance'] ].text(x, y, text)
                #self.columnAxs[self.targetFeature][ self.plotColumnD['permutationImportance'] ].set_ylabel('Mean accuracy decrease')

                # if at last row
                if self.targetN == self.nTargetFeatures-1:

                    if (len(self.targetFeatures)) == 1:

                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].set_xlabel('Features')

                    else:

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].set_xlabel('Features')

        if self.plot.rows.regressionModels.apply:

            if importanceCategory in self.plot.rows.regressionModels.columns:

                #self.columnAxs[self.regrModel][self.targetFeature][self.regrN, self.regressionModelPlotColumnD['permutationImportance'] ].bar(featureArray, permImportanceArray, yerr=errorArray, color=self.featureSymbolColor)


                if (len(self.regressorModels)) == 1:

                    self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory] ].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

                    self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory] ].tick_params(labelleft=False)

                    self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].text(.3, .95, text, ha='left', va='top',
                                                transform=self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].transAxes)

                    self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].set_ylabel(yLabel)

                else:

                    self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory] ].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

                    self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory] ].tick_params(labelleft=False)

                    self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].text(.3, .95, text, ha='left', va='top',
                                                    transform=self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].transAxes)

                    self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].set_ylabel(yLabel)

                if importanceCategory == 'featureImportance':

                    if (len(self.regressorModels)) == 1:

                        # Draw horisontal line ay y=y
                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].axhline(y=0, lw=1, c='black')

                    else:

                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].axhline(y=0, lw=1, c='black')

                # if at last row
                if self.regrN == self.nRegrModels-1:

                    if (len(self.regressorModels)) == 1:

                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].set_xlabel('Features')

                    else:

                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].set_xlabel('Features')

    def _MultCompPlotFeatureImportance(self, featureArray, importanceArray, errorArray, index, yLabel, multCompAxs):
        '''
        '''

        nnFS = self.X.shape

        text = 'nFeatures: %s' %(nnFS[1])

        if self.targetFeatureSelectionTxt != None:

            text += '\n%s' %(self.targetFeatureSelectionTxt)

        if self.agglomerateTxt != None:

            text += '\n%s' %(self.agglomerateTxt)

        if self.modelFeatureSelectionTxt != None:

            text += '\n%s' %(self.modelFeatureSelectionTxt)
            
        if 1 == 1:
            #print (self.targetFeature,index)
            #print (multCompAxs[self.targetFeature][index])
            #print (self.regressionModelPlotColumnD)
            #print (self.modelNr)
            #print ( self.multCompPlotsColumns[index] )
            #columnNr = getattr(self.multCompPlotsColumns, index)
            columnNr = self.modelNr
            #print (index)
            #print ('columnNr', columnNr)
            #print (multCompAxs[self.targetFeature][index])
            
            if (len(self.regressorModels)) == 1: # only state the column
                
                multCompAxs[self.targetFeature][index][columnNr].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

                multCompAxs[self.targetFeature][index][columnNr].tick_params(labelleft=False)

                multCompAxs[self.targetFeature][index][columnNr].text(.3, .95, text, ha='left', va='top',
                                            transform=multCompAxs[self.targetFeature][index][columnNr].transAxes)

                #multCompAxs[self.targetFeature][index][columnNr].set_ylabel(yLabel)

            else:
                                
                multCompAxs[self.targetFeature][index][self.regrN, columnNr].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

                multCompAxs[self.targetFeature][index][self.regrN, columnNr].tick_params(labelleft=False)

                multCompAxs[self.targetFeature][index][self.regrN, columnNr].text(.3, .95, text, ha='left', va='top',
                                                transform=multCompAxs[self.targetFeature][index][self.regrN, columnNr].transAxes)

                #multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_ylabel(yLabel)

            if index == 'coefficientImportance':

                if (len(self.regressorModels)) == 1:

                    # Draw horisontal line ay y=y
                    multCompAxs[self.targetFeature][index][columnNr].axhline(y=0, lw=1, c='black')

                else:

                    multCompAxs[self.targetFeature][index][self.regrN, columnNr].axhline(y=0, lw=1, c='black')

            # if at last row
            if self.regrN == self.nRegrModels-1:

                if (len(self.regressorModels)) == 1:

                    multCompAxs[self.targetFeature][index][columnNr].set_xlabel('Features')

                else:

                    multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_xlabel('Features')
                    
        # if at first column
        if columnNr == 0:

            if (len(self.regressorModels)) == 1:
                        
                multCompAxs[self.targetFeature][index][columnNr].set_ylabel(self.regrModel[0])
                multCompAxs[self.targetFeature][index][columnNr].yaxis.set_label_position("left")

            else:

                multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_ylabel(self.regrModel[0])
                multCompAxs[self.targetFeature][index][self.regrN, columnNr].yaxis.set_label_position("left")

        # if at last column
        if columnNr+1 == multCompAxs[self.targetFeature][index].shape[0]:

            if (len(self.regressorModels)) == 1:

                multCompAxs[self.targetFeature][index][columnNr].set_ylabel(yLabel)
                multCompAxs[self.targetFeature][index][columnNr].yaxis.set_label_position("right")

            else:

                multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_ylabel(yLabel)
                multCompAxs[self.targetFeature][index][self.regrN, columnNr].yaxis.set_label_position("right")





    def _FeatureImportance(self, multCompAxs):
        '''
        '''

        #Retrieve the model name and the model itself
        name,model = self.regrModel

        #Split the data into training and test subsets
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=self.modelTests.trainTest.testSize)

        #Fit the model
        model.fit(X_train, y_train)

        maxFeatures = min(self.featureImportance.reportMaxFeatures, len(self.columns))

        # Permutation importance
        n_repeats = self.featureImportance.permutationRepeats

        permImportance = permutation_importance(model, X_test, y_test, n_repeats=n_repeats)

        permImportanceMean = permImportance.importances_mean

        permImportanceStd = permImportance.importances_std

        sorted_idx = permImportanceMean.argsort()

        permImportanceArray = permImportanceMean[sorted_idx][::-1][0:maxFeatures]

        errorArray = permImportanceStd[sorted_idx][::-1][0:maxFeatures]

        featureArray = np.asarray(self.columns)[sorted_idx][::-1][0:maxFeatures]

        permImpD = {}

        for i in range(len(featureArray)):

            permImpD[featureArray[i]] = {'mean_accuracy_decrease': round(permImportanceArray[i],4),
                                         'std': round(errorArray[i],4)}
            
            #print (permImpD[featureArray[i]])

            #permImpD[featureArray[i]] = {k:(round(v,4) if isinstance(v,float) else v) for (k,v) in permImpD[featureArray[i]] }
            
        self.modelFeatureImportanceD[self.targetFeature][name]['permutationsImportance'] = permImpD

        if self.plot.singles.apply:

            title = "Permutation importance\n Feature: %s; Model: %s" %(self.targetFeature, name)

            xyLabel = ['Features', 'Mean accuracy decrease']

            pngFPN = self.imageFPND[self.targetFeature][name]['featureImportance']['permutationImportance']

            self._PlotFeatureImportanceSingles(featureArray, permImportanceArray, errorArray, title, xyLabel, pngFPN)

        if self.plot.rows.apply:

            self._PlotFeatureImportanceRows(featureArray, permImportanceArray, errorArray, 'permutationImportance', 'rel. Mean accur. decr.')

        if self.multcompplot:
      
            self._MultCompPlotFeatureImportance(featureArray, permImportanceArray, errorArray, 'permutationImportance', 'rel. Mean accur. decr.', multCompAxs)


        # Coefficient importance
        if name in ['OLS','TheilSen','Huber', "Ridge", "ElasticNet", 'logistic', 'SVR']:

            if name in ['logistic','SVR']:

                importances = model.coef_[0]

            else:

                importances = model.coef_

            absImportances = abs(importances)

            sorted_idx = absImportances.argsort()

            importanceArray = importances[sorted_idx][::-1][0:maxFeatures]

            featureArray = np.asarray(self.columns)[sorted_idx][::-1][0:maxFeatures]

            featImpD = {}

            for i in range(len(featureArray)):

                featImpD[featureArray[i]] = {'linearCoefficient': round(importanceArray[i],4)}
                
                #featImpD[featureArray[i]] = {k:(round(v,4) if isinstance(v,float) else v) for (k,v) in featImpD[featureArray[i]]}

            self.modelFeatureImportanceD[self.targetFeature][name]['featureImportance'] = featImpD

            if self.plot.singles.apply:

                title = "Linear feature coefficients\n Feature: %s; Model: %s" %(self.targetFeature, name)

                xyLabels = ['Features','Coefficient']

                pngFPN = self.imageFPND[self.targetFeature][name]['featureImportance']['regressionImportance']

                self._PlotFeatureImportanceSingles(featureArray, importanceArray, None, title, xyLabels, pngFPN)

            if self.plot.rows.apply:

                self._PlotFeatureImportanceRows(featureArray, importanceArray, None, 'coefficientImportance','rel. coef. weight')

            if self.multcompplot:
      
                self._MultCompPlotFeatureImportance(featureArray, importanceArray, None, 'coefficientImportance', 'rel. coef. weight', multCompAxs)

        elif name in ['KnnRegr','MLP', 'Cubist']:
            ''' These models do not have any feature importance to report
            '''
            pass

        else:

            featImpD = {}

            importances = model.feature_importances_

            sorted_idx = importances.argsort()

            importanceArray = importances[sorted_idx][::-1][0:maxFeatures]

            featureArray = np.asarray(self.columns)[sorted_idx][::-1][0:maxFeatures]

            if name in ['RandForRegr']:

                std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

                errorArray = std[sorted_idx][::-1][0:maxFeatures]

                for i in range(len(featureArray)):

                    featImpD[featureArray[i]] = {'MDI': round(importanceArray[i],4),
                                                 'std': round(errorArray[i],4)}
                    
                    #featImpD[featureArray[i]] = {k:(round(v,4) if isinstance(v,float) else v) for (k,v) in featImpD[featureArray[i]]}

            else:

                errorArray = None

                for i in range(len(featureArray)):

                    featImpD[featureArray[i]] = {'MDI': importanceArray[i]}
                    
                    featImpD[featureArray[i]] = {k:(round(v,4) if isinstance(v,float) else v) for (k,v) in featImpD[featureArray[i]]}

            self.modelFeatureImportanceD[self.targetFeature][name]['featureImportance'] = featImpD

            if self.plot.singles.apply:

                title = "MDI feature importance\n Feature: %s; Model: %s" %(self.targetFeature, name)

                xyLabel = ['Features', 'Mean impurity decrease']

                pngFPN = self.imageFPND[self.targetFeature][name]['featureImportance']['regressionImportance']


                #pngFPN = self.imageFPND[self.targetFeature][name]['featureImportance']['permutationImportance']

                self._PlotFeatureImportanceSingles(featureArray, importanceArray, errorArray, title, xyLabel, pngFPN)

            if self.plot.rows.apply:

                self._PlotFeatureImportanceRows(featureArray, importanceArray, errorArray, 'featureImportance', 'rel. mean impur. decr.')

    def _ManualFeatureSelector(self):
        '''
        '''

        # Reset self.columns
        self.columns = self.manualFeatureSelection.spectra

        # Create the dataframe for the sepctra
        spectraDF = self.spectraDF[ self.columns  ]

        self.manualFeatureSelectdRawBands =  self.columns
        
        self.manualFeatureSelectdDerivates = []
        
        # Create any derivative covariates requested
        
        if hasattr(self.manualFeatureSelection, 'derivatives'):
        
            if hasattr(self.manualFeatureSelection.derivatives, 'firstWaveLength'):
                
                for b in range(len(self.manualFeatureSelection.derivatives.firstWaveLength)):
        
                    bandL = [self.manualFeatureSelection.derivatives.firstWaveLength[b],
                             self.manualFeatureSelection.derivatives.lastWaveLength[b]]
    
                self.manualFeatureSelectdDerivates = bandL
    
                derviationBandDF = self.spectraDF[ bandL  ]
    
                bandFrame, bandColumn = self._SpectraDerivativeFromDf(derviationBandDF,bandL)
    
                frames = [spectraDF,bandFrame]
    
                spectraDF = pd.concat(frames, axis=1)
    
                self.columns.extend(bandColumn)

        # reset self.spectraDF
        self.spectraDF = spectraDF

    def _VarianceSelector(self):
        '''
        '''

        threshold = self.globalFeatureSelection.varianceThreshold.threshold

        #istr = 'Selected features:\nvarianceThreshold (%s)'% threshold

        #self.selectstrL.append(istr)

        # define the list of covariates to use
        #self.columnsX = [item for item in self.spectraDF.columns if item not in self.omitL]

        self.columnsX = [item for item in self.spectraDF.columns]

        # extract the covariate columns as X
        X = self.spectraDF[self.columnsX]

        #Initiate the scaler

        if self.globalFeatureSelection.scaler == 'MinMaxScaler':

            scaler = MinMaxScaler()

        scaler.fit(X)

        #Scale the data as defined by the scaler
        Xscaled = scaler.transform(X)

        #Initiate  VarianceThreshold
        select = VarianceThreshold(threshold=threshold)

        #Fit the independent variables
        select.fit(Xscaled)

        #Get the selected features from get_support as a boolean list with True or False
        selectedFeatures = select.get_support()

        #Create a list to hold discarded columns
        discardL = []

        #Create a list to hold retained columns
        self.retainL = []

        if self.verbose:

            print ('        Selecting features using VarianceThreshold, threhold =',threshold)

            print ('            Scaling function MinMaxScaler:')

        for sf in range(len(selectedFeatures)):

            if selectedFeatures[sf]:
                self.retainL.append([self.columnsX[sf],select.variances_[sf]])

            else:
                discardL.append([self.columnsX[sf],select.variances_[sf]])

        self.globalFeatureSelectedD['method'] = 'varianceThreshold'
        self.globalFeatureSelectedD['threshold'] = self.globalFeatureSelection.varianceThreshold.threshold
        #self.globalFeatureSelectedD['scaler'] = self.globalFeatureSelection.scaler
        self.globalFeatureSelectedD['nCovariatesRemoved'] = len(discardL)

        varianceSelectTxt = '%s covariates removed with %s' %(len(discardL),'VarianceThreshold')

        self.varianceSelectTxt = '%s: %s' %('VarianceThreshold',len(discardL))

        if self.verbose:

            print ('            ',varianceSelectTxt)

            if self.verbose > 1:

                #print the selected features and their variance
                print ('            Discarded features [name, (variance):')

                printL = ['%s (%.3f)'%(i[0],i[1]) for i in discardL]

                for row in printL:
                    print ('                ',row)

                print ('            Retained features [name, (variance)]:')

                printretainL = ['%s (%.3f)'%(i[0], i[1]) for i in self.retainL]

                for row in printretainL:
                    print ('                ',row)

        self.retainL = [d[0] for d in self.retainL]



        # Reset the covariate dataframe
        self.spectraDF = self.spectraDF[ self.retainL ]

    def _UnivariateSelector(self):
        '''
        '''
        nfeatures = self.X.shape[1]

        if self.targetFeatureSelection.univariateSelection.SelectKBest.apply:

            n_features = self.targetFeatureSelection.univariateSelection.SelectKBest.n_features

            if n_features >= nfeatures:

                if self.verbose:

                    infostr = '            SelectKBest: Number of features (%s) less than or equal to n_features (%s).' %(nfeatures,n_features)

                    print (infostr)

                return

        else:

            return

        # Apply SelectKBest.
        select = SelectKBest(score_func=f_regression, k=n_features)
        
        # Select and fit the independent variables, return the selected array
        X = select.fit_transform(self.X, self.y)

        # Note that the returned select.get_feature_names_out() is not a list
        self.columns = select.get_feature_names_out()

        # reset the covariates
        self.X = pd.DataFrame(X, columns=self.columns)

        self.targetFeatureSelectedD[self.targetFeature]['method'] ='SelectKBest'

        self.targetFeatureSelectedD[self.targetFeature]['nFeaturesRemoved'] = nfeatures-self.X.shape[1]

        self.targetFeatureSelectionTxt = '  %s removed %s' %( nfeatures-self.X.shape[1] ,'SelectKBest')

        if self.verbose:

            print ('\n            targetFeatureSelection:')

            print ('                ',self.targetFeatureSelectionTxt)

        if self.verbose > 1:

            print ('                Selected features: %s' %(', '.join( select.get_feature_names_out() ) ) )

    def _PermutationSelector(self):
        '''
        '''

        nfeatures = self.X.shape[1]

        n_features_to_select = self.modelFeatureSelection.RFE.n_features_to_select

        if n_features_to_select >= nfeatures:

            if self.verbose:

                infostr = '            Number of features (%s) less than or equal to n_features_to_select (%s)' %(nfeatures,n_features_to_select)

                print (infostr)

            return

        #Retrieve the model name and the model itself
        name,model = self.regrModel

        #Split the data into training and test subsets
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=self.modelTests.trainTest.testSize)

        #Fit the model
        model.fit(X_train, y_train)

        permImportance = permutation_importance(model, X_test, y_test)

        permImportanceMean = permImportance.importances_mean

        sorted_idx = permImportanceMean.argsort()

        self.columns = np.asarray(self.columns)[sorted_idx][::-1][0:n_features_to_select]

        self.X = pd.DataFrame(self.X, columns=self.columns)

        ####

        self.modelFeatureSelectedD[self.targetFeature][name]['method'] = 'PermutationSelector'

        self.modelFeatureSelectedD[self.targetFeature][name]['nFeaturesRemoved'] = nfeatures - self.X.shape[1]

        self.modelFeatureSelectionTxt = '%s feat´s removed w. %s' %( nfeatures - self.X.shape[1], 'PermutationSelector')

        if self.verbose:

            print ('\n            modelFeatureSeelction:')

            print ('                Regressor: %(m)s; Target feature: %(t)s' %{'m':name,'t':self.targetFeature})

            print ('                ',self.modelFeatureSelectionTxt)

        if self.verbose > 1:

            print ('                Selected features: %s' %(', '.join(self.columns)))

        self.modelSelectD[name] = self.columns

    def _RFESelector(self):
        '''
        '''

        nfeatures = self.X.shape[1]

        n_features_to_select = self.modelFeatureSelection.RFE.n_features_to_select

        if n_features_to_select >= nfeatures:

            if self.verbose:

                infostr = '            Number of features (%s) less than or equal to n_features_to_select (%s)' %(nfeatures,n_features_to_select)

                print (infostr)

            return

        step = self.modelFeatureSelection.RFE.step

        columns = self.X.columns

        if self.verbose:

            if self.modelFeatureSelection.RFE.CV:

                metod = 'RFECV'

                print ('\n            RFECV feature selection')

            else:

                metod = 'RFE'

                print ('\n            RFE feature selection')

        #Retrieve the model name and the model itself
        name,model = self.regrModel

        if self.modelFeatureSelection.RFE.CV:

            select = RFECV(estimator=model, min_features_to_select=n_features_to_select, step=step)

        else:

            select = RFE(estimator=model, n_features_to_select=n_features_to_select, step=step)

        select.fit(self.X, self.y)

        selectedFeatures = select.get_support()

        #Create a list to hold discarded columns
        selectL = []; discardL = []

        #print the selected features and their variance
        for sf in range(len(selectedFeatures)):
            if selectedFeatures[sf]:
                selectL.append(columns[sf])

            else:
                discardL.append(columns[sf])

        self.modelFeatureSelectedD[self.targetFeature][name]['method'] = metod

        self.modelFeatureSelectedD[self.targetFeature][name]['nFeaturesRemoved'] = len( discardL)

        self.modelFeatureSelectionTxt = '%s feat´s removed w. %s' %(len(discardL),'RFE')

        if self.verbose:

            print ('\n            modelFeatureSeelction:')

            print ('                Regressor: %(m)s; Target feature: %(t)s' %{'m':name,'t':self.targetFeature})

            print ('                ',self.modelFeatureSelectionTxt)

        if self.verbose > 1:

            print ('                Selected features: %s' %(', '.join(selectL)))

        #self.modelSelectD[name] = selectL

    def _Standardisation(self):
        """
        """
        
        self.columnsX = [item for item in self.spectraDF.columns]

        # extract the covariate columns as X
        X = self.spectraDF[self.columnsX]
        
        scaler = StandardScaler().fit(X)
        
        arrayLength = scaler.var_.shape[0]

        if self.standardisation.paretoscaling:
            #  No meancentring, scales each variable by the square root of the standard deviation

            # remove meancentring            
            scaler.mean_ = np.zeros(arrayLength)
                        
            # set var_ to its own square root
            scaler.var_ = np.sqrt(scaler.var_)
            
            # set scaling to the sqrt of the std
            scaler.scale_ = np.sqrt(scaler.var_)
            
            X1 = scaler.transform(X)            
                
        elif self.standardisation.poissonscaling:
            # No meancentring, scales each variable by the square root of the mean of the variable
            
            scaler = StandardScaler(with_mean=False).fit(X)
            
            # Set var_ to mean_
            scaler.var_ = scaler.mean_
            # set scaler to sqrt of mean_ (var_)
            scaler.scale_ = np.sqrt(scaler.var_)
            
            X1 = scaler.transform(X) 
            
        elif self.standardisation.meancentring:
            
            if self.standardisation.unitscaling:
                
                # This is a classical autoscaling or z-score normalisation
                X1 = StandardScaler().fit_transform(X)
                   
            else:
                
                # This is meancentring
                X1 = StandardScaler(with_std=False).fit_transform(X)
        
        # Reset the dataframe                
        self.spectraDF = pd.DataFrame(data=X1, columns=self.columnsX)
        
    def _PcaPreprocess(self):
        """ See https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/tutorial/astronomy/dimensionality_reduction.html
            for faster (random) algorithm
        """ 
        
        self.columnsX = [item for item in self.spectraDF.columns]

        # extract the covariate columns as X
        X = self.spectraDF[self.columnsX]
        
        # set the new covariate columns:
        
        self.columnsX = []
        
        for i in range(self.pcaPreproc.n_components):
            if i > 100:
                x = 'pc-%s' %(i)
            elif i > 10:
                x = 'pc-0%s' %(i)
            else:
                x = 'pc-00%s' %(i) 
                
            self.columnsX.append(x)
        
        pca = PCA(n_components=self.pcaPreproc.n_components)

        X_pc = pca.fit_transform(X)
                
        print(pca.components_)
        
        print(sum(pca.explained_variance_ratio_))
        
        print (self.columnsX)
        print ('X')
        print (X)
        
        print ('X_pc')
        print (X_pc)
        print (X_pc.shape)
        
        self.spectraDF = pd.DataFrame(data=X_pc, columns=self.columnsX)
        
        print (self.spectraDF)
        

                
 
            
    def _RemoveOutliers(self):
        """
        """
        
        if self.removeOutliers.contamination == 0:
            
            return
        
        #self.columnsX = [item for item in self.spectraDF.columns if item not in self.omitL]

        self.columnsX = [item for item in self.spectraDF.columns]

        # extract the covariate columns as X
        X = self.spectraDF[self.columnsX]

        iniSamples = X.shape[0]

        if self.removeOutliers.detector.lower() in ['iforest','isolationforest']:

            outlierDetector = IsolationForest(contamination=self.removeOutliers.contamination)

        elif self.removeOutliers.detector.lower() in ['ee','eenvelope','ellipticenvelope']:

            outlierDetector = EllipticEnvelope(contamination=self.removeOutliers.contamination)

        elif self.removeOutliers.detector.lower() in ['lof','lofactor','localoutlierfactor']:

            outlierDetector = LocalOutlierFactor(contamination=self.removeOutliers.contamination)

        elif self.removeOutliers.detector.lower() in ['1csvm','1c-svm','oneclasssvm']:

            outlierDetector = OneClassSVM(nu=self.removeOutliers.contamination)

        else:

            exit('unknown outlier detector')

        # The warning "X does not have valid feature names" is issued, but it is a bug and will go in next version
        yhat = outlierDetector.fit_predict(X)

        # select all rows that are not outliers
        #mask = yhat != -1

        X['yhat'] = yhat

        # Remove samples with outliers from the abudance array using the X array yhat columns
        self.abundanceDf = self.abundanceDf[ X['yhat']==1 ]

        # Keep only non-outliers in self.X
        X = X[ X['yhat']==1 ]

        # Drop the "yhat" columns for self.X
        X = X.drop(['yhat'], axis=1)

        self.spectraDF = pd.DataFrame(X)

        postSamples = X.shape[0]

        self.nOutliers = iniSamples - postSamples

        self.outliersRemovedD['method'] = self.removeOutliers.detector
        self.outliersRemovedD['nOutliersRemoved'] = self.nOutliers

        self.outlierTxt = '%s outliers removed w. %s' %(self.nOutliers,self.removeOutliers.detector)

        outlierTxt = '%s outliers removed' %(self.nOutliers)

        if self.verbose:

            print ('        ',outlierTxt)

    def _WardClustering(self, n_clusters):
        '''
        '''

        nfeatures = self.X.shape[1]

        if nfeatures < n_clusters:

            n_clusters = nfeatures

        ward = FeatureAgglomeration(n_clusters=n_clusters)

        #fit the clusters
        ward.fit(self.X, self.y)

        self.clustering =  ward.labels_

        # Get a list of bands
        bandsL =  list(self.X)

        self.aggColumnL = []

        self.aggBandL = []

        for m in range(len(ward.labels_)):

            indices = [bandsL[i] for i, x in enumerate(ward.labels_) if x == m]

            if(len(indices) == 0):

                break

            self.aggColumnL.append(indices[0])

            self.aggBandL.append( ', '.join(indices) )

        self.agglomeratedFeaturesD['method'] = 'WardClustering'

        self.agglomeratedFeaturesD['n_clusters'] = n_clusters

        self.agglomeratedFeaturesD['tuneWardClusteringApplied'] = self.featureAgglomeration.wardClustering.tuneWardClustering.apply

        agglomeratetxt = '%s input features clustered to %s covariates using  %s' %(len(self.columns),len(self.aggColumnL),self.agglomeratedFeaturesD['method'])

        self.agglomerateTxt = '%s clustered from %s to %s Feat´s' %(self.agglomeratedFeaturesD['method'], len(self.columns),len(self.aggColumnL))

        if self.verbose:

            print ('\n                ',agglomeratetxt)

            if self.verbose > 1:

                print ('                Clusters:')

                for x in range(len(self.aggColumnL)):

                    print ('                    ',self.aggBandL[x])

        # Reset the covariates (self.X)
        #X = ward.transform(self.X)

        # Reset the main dataframe
        #self.spectraDF = pd.DataFrame(X, columns=self.aggColumnL)

        # Reset the main column list
        self.columns = self.aggColumnL

        # reset the covariates
        self.X = pd.DataFrame(self.X, columns=self.aggColumnL)

    def _TuneWardClustering(self):
        ''' Determines the optimal nr of cluster
        '''
        nfeatures = self.X.shape[1]

        nClustersL = self.featureAgglomeration.wardClustering.tuneWardClustering.clusters

        nClustersL = [c for c in nClustersL if c < nfeatures]

        kfolds = self.featureAgglomeration.wardClustering.tuneWardClustering.kfolds

        cv = KFold(kfolds)  # cross-validation generator for model selection

        ridge = BayesianRidge()

        cachedir = tempfile.mkdtemp()

        mem = Memory(location=cachedir)

        ward = FeatureAgglomeration(n_clusters=4, memory=mem)

        clf = Pipeline([('ward', ward), ('ridge', ridge)])

        # Select the optimal number of parcels with grid search
        clf = GridSearchCV(clf, {'ward__n_clusters': nClustersL}, n_jobs=1, cv=cv)

        clf.fit(self.X, self.y)  # set the best parameters

        if self.verbose:

            print ('            Report for tuning Ward Clustering')

        #report the top three results
        self._ReportSearch(clf.cv_results_,3)

        #rerun with the best cluster agglomeration result
        tunedClusters = clf.best_params_['ward__n_clusters']

        if self.verbose:

            print ('                Tuned Ward clusters:', tunedClusters)

        return (tunedClusters)

    def _RandomtuningParams(self,nFeatures):
        ''' Set hyper parameters for random tuning
        '''
        self.paramDist = {}

        self.HPtuningtxt = 'Random tuning'

        # specify parameters and distributions to sample from
        name,model = self.regrModel

        if name == 'KnnRegr':

            self.paramDist[name] = {"n_neighbors": sp_randint(self.hyperParams.RandomTuning.KnnRegr.n_neigbors.min,
                                                              self.hyperParams.RandomTuning.KnnRegr.n_neigbors.max),
                          'leaf_size': sp_randint(self.hyperParams.RandomTuning.KnnRegr.leaf_size.min,
                                                              self.hyperParams.RandomTuning.KnnRegr.leaf_size.max),
                          'weights': self.hyperParams.RandomTuning.KnnRegr.weights,
                          'p': self.hyperParams.RandomTuning.KnnRegr.weights,
                          'algorithm': self.hyperParams.RandomTuning.KnnRegr.algorithm}

        elif name =='DecTreeRegr':
            # Convert 0 to None for max_depth

            max_depth = [m if m > 0 else None for m in self.hyperParams.RandomTuning.DecTreeRegr.max_depth]

            self.paramDist[name] = {"max_depth": max_depth,
                        "min_samples_split": sp_randint(self.hyperParams.RandomTuning.DecTreeRegr.min_samples_split.min,
                                                        self.hyperParams.RandomTuning.DecTreeRegr.min_samples_split.max),
                        "min_samples_leaf": sp_randint(self.hyperParams.RandomTuning.DecTreeRegr.min_samples_leaf.min,
                                                        self.hyperParams.RandomTuning.DecTreeRegr.min_samples_leaf.max)}
        elif name =='SVR':

            self.paramDist[name] = {"kernel": self.hyperParams.RandomTuning.SVR.kernel,
                                    "epsilon": self.hyperParams.RandomTuning.SVR.epsilon,
                                    "C": self.hyperParams.RandomTuning.SVR.epsilon}

        elif name =='RandForRegr':

            max_depth = [m if m > 0 else None for m in self.hyperParams.RandomTuning.RandForRegr.tuningParams.max_depth]

            max_features_max = min(self.hyperParams.RandomTuning.RandForRegr.tuningParams.max_features.max,nFeatures)

            max_features_min = min(self.hyperParams.RandomTuning.RandForRegr.tuningParams.max_features.min,nFeatures)

            print (self.hyperParams.RandomTuning.RandForRegr.tuningParams.n_estimators.min)
            
            self.paramDist[name] = {"max_depth": max_depth,
                          "n_estimators": sp_randint(self.hyperParams.RandomTuning.RandForRegr.tuningParams.n_estimators.min,
                                                              self.hyperParams.RandomTuning.RandForRegr.tuningParams.n_estimators.max),
                          "max_features": sp_randint(max_features_min,
                                                              max_features_max),
                          "min_samples_split": sp_randint(self.hyperParams.RandomTuning.RandForRegr.tuningParams.min_samples_split.min,
                                                              self.hyperParams.RandomTuning.RandForRegr.tuningParams.min_samples_split.max),
                          "min_samples_leaf": sp_randint(self.hyperParams.RandomTuning.RandForRegr.tuningParams.min_samples_leaf.min,
                                                              self.hyperParams.RandomTuning.RandForRegr.tuningParams.min_samples_leaf.max),
                          "bootstrap": self.hyperParams.RandomTuning.RandForRegr.tuningParams.bootstrap}

        elif name =='MLP':

            self.paramDist[name] = {
                        "hidden_layer_sizes": self.hyperParams.RandomTuning.MLP.hidden_layer_sizes,
                        "solver": self.hyperParams.RandomTuning.MLP.solver,
                        "alpha": sp_randint(self.hyperParams.RandomTuning.MPL.tuningParams.alpha.min,
                                    self.hyperParams.RandomTuning.MPL.tuningParams.alpha.max),
                        "max_iter": sp_randint(self.hyperParams.RandomTuning.MPL.tuningParams.max_iter.min,
                                    self.hyperParams.RandomTuning.MPL.tuningParams.max_iter.max)}

    def _ExhaustivetuningParams(self,nFeatures):
        '''
        '''

        self.HPtuningtxt = 'Exhasutive tuning'

        # specify parameters and distributions to sample from
        self.paramGrid = {}

        name,model = self.regrModel

        if name == 'KnnRegr':

            self.paramGrid[name] = [{"n_neighbors": self.hyperParams.ExhaustiveTuning.KnnRegr.tuningParams.n_neigbors,
                               'weights': self.hyperParams.ExhaustiveTuning.KnnRegr.tuningParams.weights,
                               'algorithm': self.hyperParams.ExhaustiveTuning.KnnRegr.tuningParams.algorithm,
                               'leaf_size': self.hyperParams.ExhaustiveTuning.KnnRegr.tuningParams.leaf_size,
                               'p': self.hyperParams.ExhaustiveTuning.KnnRegr.tuningParams.p}
                               ]
        elif name =='DecTreeRegr':
            max_depth = [m if m > 0 else None for m in self.hyperParams.ExhaustiveTuning.DecTreeRegr.tuningParams.max_depth]

            self.paramGrid[name] = [{
                                "splitter": self.hyperParams.ExhaustiveTuning.DecTreeRegr.tuningParams.splitter,
                                "max_depth": self.hyperParams.ExhaustiveTuning.DecTreeRegr.tuningParams.max_depth,
                                "min_samples_split": self.hyperParams.ExhaustiveTuning.DecTreeRegr.tuningParams.min_samples_split,
                                "min_samples_leaf": self.hyperParams.ExhaustiveTuning.DecTreeRegr.tuningParams.min_samples_leaf}]

        elif name =='SVR':
            self.paramGrid[name] = [{"kernel": self.hyperParams.ExhaustiveTuning.SVR.tuningParams.kernel,
                                "epsilon": self.hyperParams.ExhaustiveTuning.SVR.tuningParams.epsilon,
                                "C": self.hyperParams.ExhaustiveTuning.SVR.tuningParams.C
                              }]

        elif name =='RandForRegr':
            max_depth = [m if m > 0 else None for m in self.hyperParams.ExhaustiveTuning.RandForRegr.tuningParams.max_depth]

            self.paramGrid[name] = [{
                            "max_depth": max_depth,
                          "n_estimators": self.hyperParams.ExhaustiveTuning.RandForRegr.tuningParams.n_estimators,
                          "min_samples_split": self.hyperParams.ExhaustiveTuning.RandForRegr.tuningParams.min_samples_split,
                          "min_samples_leaf": self.hyperParams.ExhaustiveTuning.RandForRegr.tuningParams.min_samples_leaf,
                          "bootstrap": self.hyperParams.ExhaustiveTuning.RandForRegr.tuningParams.bootstrap}]

        elif name =='MLP':
            self.paramGrid[name] = [{
                        "hidden_layer_sizes": self.hyperParams.ExhaustiveTuning.MLP.tuningParams.hidden_layer_sizes,
                        "solver": self.hyperParams.ExhaustiveTuning.MLP.tuningParams.solver,
                        "alpha": self.hyperParams.ExhaustiveTuning.MLP.tuningParams.alpha,
                        "max_iter": self.hyperParams.ExhaustiveTuning.MLP.tuningParams.max_iter}]

    def _RandomTuning(self):
        '''
        '''

        #Retrieve the model name and the model itself
        name,mod = self.regrModel

        nFeatures = self.X.shape[1]

        # Get the tuning parameters
        self._RandomtuningParams(nFeatures)

        if self.verbose:

            print ('\n                HyperParameter random tuning:')

            print ('                    ',name, self.paramDist[name])

        search = RandomizedSearchCV(mod, param_distributions=self.paramDist[name],
                                           n_iter=self.params.hyperParameterTuning.nIterSearch)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=(1-self.params.hyperParameterTuning.fraction))

        search.fit(X_train, y_train)

        resultD = self._ReportSearch(search.cv_results_,self.params.hyperParameterTuning.n_best_report)

        self.tunedHyperParamsD[self.targetFeature][name] = resultD

        # Set the hyperParameters to the best result
        for key in resultD[1]['hyperParameters']:

            self.jsonparamsD['regressionModels'][name]['hyperParams'][key] = resultD[1]['hyperParameters'][key]

    def _ExhaustiveTuning(self):
        '''
        '''

        #Retrieve the model name and the model itself
        name,mod = self.regrModel

        nFeatures = self.X.shape[1]

        # Get the tuning parameters
        self._ExhaustivetuningParams(nFeatures)

        if self.verbose:

            print ('\n                HyperParameter exhaustive tuning:')

            print ('                    ',name, self.paramGrid[name])

        search = GridSearchCV(mod, param_grid=self.paramGrid[name])

        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=(1-self.params.hyperParameterTuning.fraction))

        search.fit(X_train, y_train)

        resultD = self._ReportSearch(search.cv_results_,self.params.hyperParameterTuning.n_best_report)

        self.tunedHyperParamsD[self.targetFeature][name] = resultD

        # Set the hyperParameters to the best result
        for key in resultD[1]['hyperParameters']:

            self.jsonparamsD['regressionModels'][name]['hyperParams'][key] = resultD[1]['hyperParameters'][key]

    def _ReportRegModelParams(self):
        '''
        '''

        print ('            Model hyper-parameters:')

        for model in self.regressorModels:

            #Retrieve the model name and the model itself
            modelname,modelhyperparams = model

            print ('                name', modelname, modelhyperparams.get_params())

    def _ReportSearch(self, results, n_top=3):
        '''
        '''

        resultD = {}
        for i in range(1, n_top + 1):

            resultD[i] = {}

            candidates = np.flatnonzero(results['rank_test_score'] == i)

            for candidate in candidates:

                resultD[i]['mean_test_score'] = results['mean_test_score'][candidate]

                resultD[i]['std'] = round(results['std_test_score'][candidate],4)

                resultD[i]['std'] = round(results['std_test_score'][candidate],4)

                resultD[i]['hyperParameters'] = results['params'][candidate]

                if self.verbose:

                    print("                    Model with rank: {0}".format(i))

                    print("                    Mean validation score: {0:.3f} (std: {1:.3f})".format(
                          results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))

                    print("                    Parameters: {0}".format(results['params'][candidate]))

                    print("")

        return resultD

class MachineLearningModel(Obj, RegressionModels):
    ''' MAchine Learning model of feature propertie from spectra
    '''

    def __init__(self,paramD):
        """ Convert input parameters from nested dict to nested class object

            :param dict paramD: parameters
        """

        # convert the input parameter dict to class objects
        Obj.__init__(self,paramD)

        # initiate the regression models
        RegressionModels.__init__(self)

        self.paramD = paramD

        # Set class object default data if required
        self._SetModelDefaults()

        # Deep copy parameters to a new object class called params
        self.params = deepcopy(self)

        # Drop the plot and figure settings from paramD
        paramD.pop('plot')

        # Deep copy the parameters to self.soillineD
        self.plotD = deepcopy(paramD)
                
    def _SetSrcFPNs(self, rootFP, dstRootFP, sourcedatafolder):
        ''' Set source file paths and names
        '''

        # All OSSL data are download as a zipped subfolder with data given standard names as of below
               
        # if the path to rootFP starts with a dot '.' (= self) then use the default rootFP 
        if self.input.jsonSpectraDataFilePath[0] == '.':
            
            removeStart = 1
            
            if self.input.jsonSpectraDataFilePath[1] in ['/']:
                
                removeStart = 2
            
            dataSubFP = self.input.jsonSpectraDataFilePath[removeStart: len(self.input.jsonSpectraDataFilePath)]
               
            jsonSpectraDataFilePath = os.path.join(dstRootFP, dataSubFP)
            
        else:
            
            jsonSpectraDataFilePath = self.input.jsonSpectraDataFilePath
            
        if self.input.jsonSpectraParamsFilePath[0] == '.':
            
            removeStart = 1
            
            if self.input.jsonSpectraParamsFilePath[1] in ['/']:
                
                removeStart = 2
            
            paramSubFP = self.input.jsonSpectraParamsFilePath[removeStart: len(self.input.jsonSpectraParamsFilePath)]
               
            jsonSpectraParamsFilePath = os.path.join(dstRootFP, paramSubFP)
            

        else:
        
            jsonSpectraParamsFilePath = self.input.jsonSpectraParamsFilePath
            
        if not os.path.exists(jsonSpectraDataFilePath):
            
            exitStr = 'Data file not found: %s ' %(jsonSpectraDataFilePath)
            
            exit(exitStr)

        if not os.path.exists(jsonSpectraParamsFilePath):
            
            exitStr = 'Param file not found: %s ' %(jsonSpectraParamsFilePath)
            
            exit(exitStr)
            
        self.dataFPN = jsonSpectraDataFilePath
        
        # Open and load JSON data file
        with open(jsonSpectraDataFilePath) as jsonF:

            self.jsonSpectraData = json.load(jsonF)

        # Open and load JSON parameter file
        with open(jsonSpectraParamsFilePath) as jsonF:

            self.jsonSpectraParams = json.load(jsonF)
            
    def _SetColorRamp(self,n):
        ''' Slice predefined colormap to discrete colors for each band
        '''

        # Set colormap to use for plotting
        cmap = plt.get_cmap(self.plot.colorramp)

        # Segmenting colormap to the number of bands
        self.slicedCM = cmap(np.linspace(0, 1, n))

    def _GetAbundanceData(self):
        '''
        '''

        # Get the list of substances included in this dataset

        substanceColumns = self.jsonSpectraParams['labData']

        #substanceColumns = self.jsonSpectraParams['targetFeatures']

        substanceOrderD = {}

        for substance in substanceColumns:

            substanceOrderD[substance] = substanceColumns.index(substance)

        n = 0

        # Loop over the samples
        for sample in self.jsonSpectraData['spectra']:

            substanceL = [None] * len(substanceColumns)

            for abundance in sample['abundances']:

                substanceL[ substanceOrderD[abundance['substance']] ] = abundance['value']

            if n == 0:

                abundanceA = np.asarray(substanceL, dtype=float)

            else:

                abundanceA = np.vstack( (abundanceA, np.asarray(substanceL, dtype=float) ) )

            n += 1

        self.abundanceDf = pd.DataFrame(data=abundanceA, columns=substanceColumns)

        self.transformD = {}
        
        # Do any transformation requested
        for column in substanceColumns:
            
            self.transformD[column] = 'linear'
            
            if hasattr(self.params.targetFeatureTransform, column):
                
                targetTransform = getattr(self.params.targetFeatureTransform, column)
                
                if targetTransform.log:
                    
                    self.abundanceDf[column] = np.log(self.abundanceDf[column])
                    
                    self.transformD[column] = 'log'
                    
                elif targetTransform.sqrt:
                    
                    self.abundanceDf[column] = np.sqrt(self.abundanceDf[column])
                    
                    self.transformD[column] = 'sqrt'
                    
                elif targetTransform.reciprocal:
                    
                    self.abundanceDf[column] = np.reciprocal(self.abundanceDf[column])
                    
                    self.transformD[column] = 'reciprocal'
                    
                elif targetTransform.boxcox:
                    
                    pt = PowerTransformer(method='box-cox')
                    
                    try:
                    
                        pt.fit( np.atleast_2d(self.abundanceDf[column]) )
                        
                        print(pt.lambdas_)
                        
                        print(pt.transform( np.atleast_2d(self.abundanceDf[column]) ))
                        
                        X = pt.transform( np.atleast_2d(self.abundanceDf[column]) )
                        
                        print (X)
                        
                        self.abundanceDf[column], boxcoxLambda = boxcox(self.abundanceDf[column])
                                            
                        self.transformD[column] = 'boxcox'
                        
                        self.abundanceDf[column]
                        
                        print ( self.abundanceDf[column] )
                        
                        SNULLE
                        
                    except:
                        
                        print('cannot box-cox transform')
                elif targetTransform.yeojohnson:
                    
                    pt = PowerTransformer()
                    
                    pt.fit(np.atleast_2d(self.abundanceDf[column]))
                    
         
                    
                    print(pt.lambdas_)
                    
                    print(pt.transform( np.atleast_2d(self.abundanceDf[column]) ) )
                    
                    X = pt.transform( np.atleast_2d(self.abundanceDf[column]) )
                    
                    print (X)
                    
                    BALLE
                    

                        
                elif targetTransform.quantile:
                    
                    pass
                    
                    #X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
                    #qt = QuantileTransformer(n_quantiles=10, random_state=0)
                    
            if hasattr(self.params.targetFeatureStandardise, column):
                    
                standardise =  getattr(self.params.targetFeatureStandardise, column)
                
                if standardise:
                    
                    #X = scale(self.abundanceDf[column], axis=0, with_mean=True, with_std=True, copy=True)
                    
                    #print (X)
                    
                    self.abundanceDf[column] = (self.abundanceDf[column] - np.mean(self.abundanceDf[column]) ) / np.std(self.abundanceDf[column])
                    
                    #print (self.abundanceDf[column])
                            
    def _StartStepSpectra(self, pdSpectra, startwl, stopwl, stepwl):
        '''
        '''

        wlmin = pdSpectra['wl'].min()
        wlmax = pdSpectra['wl'].max()
        step = (wlmax-wlmin)/(pdSpectra.shape[0])

        startindex = (startwl-wlmin)/step

        stopindex = (stopwl-wlmin)/step

        stepindex = stepwl/step

        indexL = []; iL = []

        i = 0

        while True:

            if i*stepindex+startindex > stopindex+1:

                break

            indexL.append(int(i*stepindex+startindex))
            iL.append(i)

            i+=1

        df = pdSpectra.iloc[indexL]

        return df, indexL[0]

    def _SpectraDerivativeFromDf(self,dataFrame,columns):
        ''' Create spectral derivates
        '''

        # Get the derivatives
        spectraDerivativeDF = dataFrame.diff(axis=1, periods=1)

        # Drop the first column as it will have only NaN
        spectraDerivativeDF = spectraDerivativeDF.drop(columns[0], axis=1)

        # Reset columns to integers
        columns = [int(i) for i in columns]

        # Create the derivative columns
        derivativeColumns = ['d%s' % int((columns[i-1]+columns[i])/2) for i in range(len(columns)) if i > 0]

        # Replace the columns
        spectraDerivativeDF.columns = derivativeColumns

        return spectraDerivativeDF, derivativeColumns

    def _GetBandData(self):
        ''' Read json data into numpy array and convert to pandas dataframe
        '''
        self.varianceSelectTxt = None; self.outlierTxt = None
        self.targetFeatureSelectionTxt = None; self.agglomerateTxt = None
        self.modelFeatureSelectionTxt = None

        # Use the wavelength as column headers
        self.columns = self.jsonSpectraData['waveLength']

        # Convert the column headers to strings
        self.columns = [str(c) for c in self.columns]

        n = 0

        # Loop over the spectra
        for sample in self.jsonSpectraData['spectra']:

            if n == 0:

                spectraA = np.asarray(sample['signalMean'])

            else:

                spectraA = np.vstack( (spectraA, np.asarray(sample['signalMean']) ) )

            n += 1

        self.spectraDF = pd.DataFrame(data=spectraA, columns=self.columns)

        if self.derivatives.apply:

            spectraDerivativeDF,derivativeColumns = self._SpectraDerivativeFromDf(self.spectraDF, self.columns)

            if self.derivatives.join:

                frames = [self.spectraDF, spectraDerivativeDF]

                self.spectraDF = pd.concat(frames, axis=1)

                self.columns.extend(derivativeColumns)

            else:

                self.spectraDF = spectraDerivativeDF

                self.columns = derivativeColumns

        self.originalColumns = self.columns

    def _SetSubPlots(self):
        '''
        '''

        #regressionModelTitleTranslateD = {'permutationImportance': 'Permutation importance', 'Kfold': 'Kfold model','trainTest': 'train/test model'}

        #targetFeatureTitleTranslateD = {'permutationImportance': 'Permutation importance', 'Kfold': 'Kfold model','trainTest': 'train/test model'}

        if self.plot.rows.apply:

            self.nRegrModels = len(self.regressorModels)

            self.nTargetFeatures = len(self.targetFeatures)

            self.columnFig = {}

            self.columnAxs = {}

            if self.plot.rows.targetFeatures.apply:

                self.targetFeaturePlotColumnD = {}

                for c, col in enumerate(self.plot.rows.targetFeatures.columns):

                    self.targetFeaturePlotColumnD[col] = c

                self.targetFeaturesFigCols = len(self.plot.rows.targetFeatures.columns)

                # Set the figure size

                xadd = self.plot.rows.targetFeatures.figSize.xadd

                if  xadd == 0:

                    xadd = self.plot.rows.subFigSize.xadd

                if self.plot.rows.targetFeatures.figSize.x == 0:

                    figSizeX = self.plot.rows.subFigSize.x * self.targetFeaturesFigCols + xadd

                else:

                    figSizeX = self.plot.rows.targetFeatures.figSize.x

                yadd = self.plot.rows.targetFeatures.figSize.yadd

                if  yadd == 0:

                    yadd = self.plot.rows.subFigSize.yadd

                if self.plot.rows.targetFeatures.figSize.y == 0:

                    figSizeY = self.plot.rows.subFigSize.y * self.nTargetFeatures + yadd

                else:

                    figSizeY = self.plot.rows.targetFeatures.figSize.y

                # Create column plots for individual targetFeatures, with rows showing different regressors
                for regrModel in self.regressorModels:

                    self.columnFig[regrModel[0]], self.columnAxs[regrModel[0]] = plt.subplots( self.nTargetFeatures, self.targetFeaturesFigCols, figsize=(figSizeX, figSizeY) )

                    if self.plot.tightLayout:

                        self.columnFig[regrModel[0]].tight_layout()

                    # Set title
                    suptitle = "Regressor: %s, %s (rows=target features)\n" %(regrModel[0], self.hyperParamtxt)

                    suptitle += '%s input features' %(len(self.originalColumns))

                    # Set subplot wspace and hspace
                    if self.plot.rows.regressionModels.hwspace.wspace:

                        self.columnFig[regrModel[0]].subplots_adjust(wspace=self.plot.rows.regressionModels.hwspace.wspace)

                    if self.plot.rows.regressionModels.hwspace.hspace:

                        self.columnFig[regrModel[0]].subplots_adjust(hspace=self.plot.rows.regressionModels.hwspace.hspace)

                    if self.varianceSelectTxt != None:

                        suptitle += ', %s' %(self.varianceSelectTxt)

                    if self.outlierTxt != None:

                        suptitle +=  ', %s' %(self.outlierTxt)

                    self.columnFig[regrModel[0]].suptitle(  suptitle )

                    for r,rows in enumerate(self.targetFeatures):

                        for c,cols in enumerate(self.plot.rows.targetFeatures.columns):

                            # Set subplot titles:
                            if 'Importance' in cols:

                                if r == 0:

                                    title = '%s' %( cols.replace('Importance', ' Importance'))

                                    if (len(self.targetFeatures)) == 1:

                                        self.columnAxs[ regrModel[0] ][c].set_title(title)

                                    else:

                                        self.columnAxs[ regrModel[0] ][r,c].set_title(title)

                            else:

                                title = '%s %s' %( self.paramD['targetFeatureSymbols'][rows]['label'], cols)

                                if (len(self.targetFeatures)) == 1:

                                    self.columnAxs[ regrModel[0] ][c].set_title(title)

                                else:

                                    self.columnAxs[ regrModel[0] ][r,c].set_title(title)

            if self.plot.rows.regressionModels.apply:

                self.regressionModelPlotColumnD = {}

                for c, col in enumerate(self.plot.rows.regressionModels.columns):

                    self.regressionModelPlotColumnD[col] = c

                self.regressionModelFigCols = len(self.plot.rows.regressionModels.columns)

                # Set the figure size

                xadd = self.plot.rows.regressionModels.figSize.xadd

                if  xadd == 0:

                    xadd = self.plot.rows.subFigSize.xadd

                if self.plot.rows.regressionModels.figSize.x == 0:

                    figSizeX = self.plot.rows.subFigSize.x * self.regressionModelFigCols + xadd

                else:

                    figSizeX = self.plot.rows.regressionModels.figSize.x

                yadd = self.plot.rows.regressionModels.figSize.yadd

                if  yadd == 0:

                    yadd = self.plot.rows.subFigSize.yadd

                if self.plot.rows.regressionModels.figSize.y == 0:

                    figSizeY = self.plot.rows.subFigSize.y * self.nRegrModels + yadd

                else:

                    figSizeY = self.plot.rows.regressionModels.figSize.x

                # Create column plots for individual regressionModels, with rows showing different regressors
                for targetFeature in self.targetFeatures:

                    self.columnFig[targetFeature], self.columnAxs[targetFeature] = plt.subplots( self.nRegrModels, self.regressionModelFigCols, figsize=(figSizeX, figSizeY))

                    # ERROR If only one regressionModle then r == NONE

                    if self.plot.tightLayout:

                        self.columnFig[targetFeature].tight_layout()

                    # Set subplot wspace and hspace
                    if self.plot.rows.targetFeatures.hwspace.wspace:

                        self.columnFig[targetFeature].subplots_adjust(wspace=self.plot.rows.targetFeatures.hwspace.wspace)

                    if self.plot.rows.targetFeatures.hwspace.hspace:

                        self.columnFig[targetFeature].subplots_adjust(hspace=self.plot.rows.targetFeatures.hwspace.hspace)

                    label = self.paramD['targetFeatureSymbols'][targetFeature]['label']

                    suptitle = "Target: %s, %s (rows=regressors)\n" %(label, self.hyperParamtxt )

                    suptitle += '%s input features' %(len(self.originalColumns))

                    if self.varianceSelectTxt != None:

                        suptitle += ', %s' %(self.varianceSelectTxt)

                    if self.outlierTxt != None:

                        suptitle +=  ', %s' %(self.outlierTxt)

                    # Set suotitle
                    self.columnFig[targetFeature].suptitle( suptitle )

                    # Set subplot titles:
                    for r,rows in enumerate(self.regressorModels):

                        for c,cols in enumerate(self.plot.rows.regressionModels.columns):

                            #title = '%s %s' %(rows[0], cols)

                            # Set subplot titles:
                            if 'Importance' in cols:

                                if r == 0:

                                    title = '%s' %( cols.replace('Importance', ' Importance'))

                                    if (len(self.regressorModels)) == 1:

                                        self.columnAxs[targetFeature][c].set_title( title )

                                    else:

                                        self.columnAxs[targetFeature][r,c].set_title( title )

                            else:

                                title = '%s %s' %(rows[0], cols)
                                #title = '%s ' %( self.paramD['targetFeatureSymbols'][rows]['label'], cols)
                                if (len(self.regressorModels)) == 1:

                                    self.columnAxs[targetFeature][c].set_title( title )

                                else:

                                    self.columnAxs[targetFeature][r,c].set_title( title )

                            #regressionModelTitleTranslateD = {'permutationImportance': 'Permutation importance', 'Kfold': 'Kfold model','trainTest': 'train/test model'}

    def _SetDstFPNs(self):
        ''' Set destination file paths and names
        '''

        FP,FN = os.path.split(self.dataFPN)

        FN = os.path.splitext(FN)[0]

        modelFP = os.path.join(FP,'mlmodel')

        if not os.path.exists(modelFP):

            os.makedirs(modelFP)

        modelresultFP = os.path.join(modelFP,'json')

        if not os.path.exists(modelresultFP):

            os.makedirs(modelresultFP)

        pickleFP = os.path.join(modelFP,'pickle')

        if not os.path.exists(pickleFP):

            os.makedirs(pickleFP)

        modelimageFP = os.path.join(modelFP,'images')

        if not os.path.exists(modelimageFP):

            os.makedirs(modelimageFP)
            
        # prefix tells if the modeling is done from manual setting, raw spectra, derivatives or both
        if self.manualFeatureSelection.apply:
            
            prefix = 'manual_'
            
        else:
            
            prefix =  'spectra_'
               
            if self.derivatives.apply:
    
                if self.derivatives.join:
                    
                    prefix =  'spectra+derivative_'
        
                else:
        
                    prefix =  'derivative_'
                
        # if prefix is given it will be added to all output files
        if len(self.output.prefix) > 0 and self.output.prefix[len(self.output.prefix)-1] != '_':

            prefix = '%s_%s' %(self.output.prefix, prefix)

        else:

            prefix = self.output.prefix
            
        summaryJsonFN = '%s%s_summary.json' %(prefix, self.name)

        self.summaryJsonFPN = os.path.join(modelresultFP,summaryJsonFN)

        regrJsonFN = '%s%s_results.json' %(prefix, self.name)

        self.regrJsonFPN = os.path.join(modelresultFP,regrJsonFN)

        paramJsonFN = '%s%s_params.json' %(prefix,self.name)

        self.paramJsonFPN = os.path.join(modelresultFP,paramJsonFN)

        self.imageFPND = {}

        # the picke files save the regressor models for later use
        self.trainTestPickleFPND = {}

        self.KfoldPickleFPND = {}
        
        # loop over targetfeatures
        for targetFeature in self.paramD['targetFeatures']:

            self.imageFPND[targetFeature] = {}

            self.trainTestPickleFPND[targetFeature] = {}; self.KfoldPickleFPND[targetFeature] = {}

            for regmodel in self.paramD['regressionModels']:

                trainTestPickleFN = '%s%s_%s_%s_trainTest.xsp'    %(prefix,'modelid',targetFeature, regmodel)

                KfoldPickleFN = '%s%s_%s_%s_Kfold.xsp'    %(prefix,'modelid',targetFeature, regmodel)

                self.trainTestPickleFPND[targetFeature][regmodel] = os.path.join(pickleFP, trainTestPickleFN)

                self.KfoldPickleFPND[targetFeature][regmodel] = os.path.join(pickleFP, KfoldPickleFN)

                self.imageFPND[targetFeature][regmodel] = {}

                if self.featureImportance.apply:

                    self.imageFPND[targetFeature][regmodel]['featureImportance'] = {}

                    imgFN = '%s%s_%s-model_permut-imp.png'    %(prefix,targetFeature, regmodel)

                    self.imageFPND[targetFeature][regmodel]['featureImportance']['permutationImportance'] = os.path.join(modelimageFP, imgFN)

                    imgFN = '%s%s_%s-model_feat-imp.png'    %(prefix,targetFeature, regmodel)

                    self.imageFPND[targetFeature][regmodel]['featureImportance']['regressionImportance'] = os.path.join(modelimageFP, imgFN)

                if self.modelTests.trainTest.apply:

                    imgFN = '%s%s_%s-model_tt-result.png'    %(prefix,targetFeature, regmodel)

                    self.imageFPND[targetFeature][regmodel]['trainTest'] = os.path.join(modelimageFP, imgFN)

                if self.modelTests.Kfold.apply:

                    imgFN = '%s%s_%s-model_kfold-result.png'    %(prefix,targetFeature, regmodel)

                    self.imageFPND[targetFeature][regmodel]['Kfold'] = os.path.join(modelimageFP, imgFN)

            # Set multi row-image file names, per targetfeature
            imgFN = '%s%s-multi-results.png'    %(prefix, targetFeature)

            self.imageFPND[targetFeature]['allmodels'] = os.path.join(modelimageFP, imgFN)

        for regmodel in self.paramD['regressionModels']:

            self.imageFPND[regmodel] = {}

            # Set multi row-image file names, per regression model
            imgFN = '%s%s-multi-results.png'    %(prefix, regmodel)

            self.imageFPND[regmodel]['alltargets'] = os.path.join(modelimageFP, imgFN)


    def _DumpJson(self):
        '''
        '''

        resultD = {}; summaryD = {}; self.multCompSummaryD = {}
        
        for targetFeature in self.targetFeatures:
        
            self.multCompSummaryD[targetFeature] = {}
        
        #resultD['targetFeatures'] = self.transformD
        
        #self.summaryD['targetFeatures'] = self.transformD

        resultD['originalInputColumns'] = len(self.originalColumns)
        
            
        if self.standardisation.apply:
            
            resultD['standardisation'] = True
            
        if self.pcaPreproc.apply:
            
            resultD['pcaPreproc'] = True

        if self.removeOutliers.apply or self.globalFeatureSelection.apply or self.featureAgglomeration.apply:

            resultD['globalTweaks']= {}

            if self.removeOutliers.apply:

                resultD['globalTweaks']['removeOutliers'] = self.outliersRemovedD

            if self.globalFeatureSelection.apply:

                resultD['globalTweaks']['globalFeatureSelection'] = self.globalFeatureSelectedD

            if self.featureAgglomeration.apply:

                resultD['globalTweaks']['featureAgglomeration'] = self.agglomeratedFeaturesD

        if self.manualFeatureSelection.apply:

            resultD['manualFeatureSelection'] = True

        if self.targetFeatureSelection.apply:

            resultD['targetFeatureSelection'] = self.targetFeatureSelectedD

        if self.modelFeatureSelection.apply:

            resultD['modelFeatureSelection'] = self.modelFeatureSelectedD

        if self.featureImportance:

            resultD['featureImportance'] = self.modelFeatureImportanceD

        if self.hyperParameterTuning.apply:

            resultD['hyperParameterTuning'] = {}

            if self.hyperParameterTuning.randomTuning.apply:

                # Set the results from the hyperParameter Tuning
                resultD['hyperParameterTuning']['randomTuning'] = self.tunedHyperParamsD

            if self.hyperParameterTuning.exhaustiveTuning.apply:

                # Set the results from the hyperParameter Tuning
                resultD['hyperParameterTuning']['exhaustiveTuning'] = self.tunedHyperParamsD

        # Add the finally selected bands

        resultD['appliedModelingFeatures'] = self.finalFeatureLD

        # Add the final model results
        if self.modelTests.apply:

            resultD['modelResults'] = {}
            
            summaryD['modelResults'] = {}

            if self.modelTests.trainTest.apply:

                resultD['modelResults']['trainTest'] = self.trainTestResultD
                
                summaryD['modelResults']['trainTest'] = self.trainTestSummaryD
                
                for targetFeature in self.targetFeatures:
                
                    self.multCompSummaryD[targetFeature]['trainTest'] = self.trainTestSummaryD[targetFeature]
                
                    self.multCompSummaryD[targetFeature]['parameters'] = self.paramJsonFPN
                    
                    self.multCompSummaryD[targetFeature]['results'] = self.regrJsonFPN
                    
            if self.modelTests.Kfold.apply:

                resultD['modelResults']['Kfold'] = self.KfoldResultD
                
                summaryD['modelResults']['Kfold'] = self.KfoldSummaryD
                
                for targetFeature in self.targetFeatures:
                
                    self.multCompSummaryD[targetFeature]['Kfold'] = self.KfoldSummaryD[targetFeature]
                    
                    self.multCompSummaryD[targetFeature]['parameters'] = self.paramJsonFPN
                    
                    self.multCompSummaryD[targetFeature]['results'] = self.regrJsonFPN

        #pp = pprint.PrettyPrinter(indent=2)
        #pp.pprint(resultD)

        jsonF = open(self.regrJsonFPN, "w")
        
        json.dump(resultD, jsonF, indent = 2)

        jsonF = open(self.paramJsonFPN, "w")

        json.dump(self.paramD, jsonF, indent = 2)
        
        jsonF = open(self.summaryJsonFPN, "w")

        json.dump(summaryD, jsonF, indent = 2)

    def _PlotTitleTextn(self, titleSuffix,plotskipstep):
        ''' Set plot title and annotation

            :param str titleSuffix: amendment to title

            :returns: x-axis label
            :rtype: str

            :returns: y-axis label
            :rtype: str

            :returns: title
            :rtype: str

            :returns: text
            :rtype: str
        '''

        # Set title
        title = self.name

        # set the text
        text = self.plot.text.text

        # Add the bandwidth
        if self.plot.text.bandwidth:

            bandwidth = (max(self.columns)- min(self.columns))/(len(self.columns)-1)

            text += '\nbandwidth=%s nm' %( bandwidth )

        # Add number of samples to text
        if self.plot.text.samples:

            text += '\nnspectra=%s; nbands=%s' %( self.spectraDF.shape[0],len(self.columns))
            text += '\nshowing every %s spectra' %( plotskipstep )

        yLabel = self.plot.rawaxislabel.x

        xLabel = self.plot.rawaxislabel.y

        return (xLabel, yLabel, title, text)

    def _PilotModeling(self,rootFP,sourcedatafolder,dstRootFP, multCompFig, multCompAxs):
        ''' Steer the sequence of processes for modeling spectra data in json format
        '''

        if len(self.targetFeatures) == 0:

            exit('Exiting - you have to set at least 1 target feature')

        if len(self.regressorModels) == 0:

            exit('Exiting - you have to set at least 1 regressor')
            
        # Set the source file names
        self._SetSrcFPNs(rootFP, dstRootFP, sourcedatafolder)
        
        # set the destination file names
        self._SetDstFPNs()

        # Get the band data as self.spectraDF
        self._GetBandData()

        # Get and add the abundance data
        self._GetAbundanceData()

        self.hyperParamtxt = "hyper-param tuning: None"

        if self.hyperParameterTuning.apply:

            if self.hyperParameterTuning.exhaustiveTuning.apply:

                hyperParameterTuning = 'ExhaustiveTuning'

                self.tuningParamD = ReadModelJson(self.input.hyperParameterExhaustiveTuning)

                self.hyperParamtxt = "hyper-param tuning: grid search"

            elif self.hyperParameterTuning.randomTuning.apply:

                hyperParameterTuning = 'RandomTuning'

                self.tuningParamD = ReadModelJson(self.input.hyperParameterRandomTuning)

                self.hyperParamtxt = "hyper-param tuning: random"

            else:

                errorStr = 'Hyper parameter tuning requested, but no method assigned'

                exit(errorStr)

            self.hyperParams = Obj(self.tuningParamD )

        # Set the dictionaries to hold the model results
        self.trainTestResultD = {}; self.KfoldResultD  = {}; self.tunedHyperParamsD = {}
        self.globalFeatureSelectedD = {}; self.outliersRemovedD = {};
        self.agglomeratedFeaturesD = {}; self.targetFeatureSelectedD = {}
        self.modelFeatureSelectedD = {}; self.modelFeatureImportanceD = {}
        self.finalFeatureLD = {}
        self.trainTestSummaryD = {}; self.KfoldSummaryD  = {};

        # Create the subDicts for all model + target related presults
        for targetFeature in self.targetFeatures:

            self.tunedHyperParamsD[targetFeature] = {}; self.trainTestResultD[targetFeature] = {}
            self.KfoldResultD[targetFeature] = {}; self.modelFeatureSelectedD[targetFeature] = {}
            self.targetFeatureSelectedD[targetFeature] = {}; self.modelFeatureImportanceD[targetFeature] = {}
            self.finalFeatureLD[targetFeature] = {}
            self.trainTestSummaryD[targetFeature] = {}; self.KfoldSummaryD[targetFeature]  = {};

            for regModel in self.jsonparamsD['regressionModels']:

                if self.paramD['regressionModels'][regModel]['apply']:

                    self.trainTestResultD[targetFeature][regModel] = {}
                    self.KfoldResultD[targetFeature][regModel] = {}
                    self.modelFeatureSelectedD[targetFeature][regModel] = {}
                    self.modelFeatureImportanceD[targetFeature][regModel] = {}
                    self.finalFeatureLD[targetFeature][regModel] = {}
                    self.trainTestSummaryD[targetFeature][regModel] = {} 
                    self.KfoldSummaryD[targetFeature][regModel] = {}
                    
                    # Set the transformation to the output dict      
                    self.trainTestSummaryD[targetFeature]['transform'] = self.transformD[targetFeature]
                    
                    self.KfoldSummaryD[targetFeature]['transform'] = self.transformD[targetFeature]
                    
                    self.trainTestResultD[targetFeature]['transform'] = self.transformD[targetFeature]
 
                    self.KfoldResultD[targetFeature]['transform'] = self.transformD[targetFeature]
                        
                    if self.paramD['hyperParameterTuning']['apply'] and self.tuningParamD[hyperParameterTuning][regModel]['apply']:

                        self.tunedHyperParamsD[targetFeature][regModel] = {}

        
        
        # standardisation can do both meancetnring or -score normalisation            
        if self.scattercorrection.apply:
       
            self.spectraDF, self.scatCorrScaler, self.scaCcorrMeanSpectra = SpectraScatterCorrectionFromDf(self.spectraDF,self.scattercorrection)
            
        # standardisation can do both meancetnring or z-score normalisation            
        if self.standardisation.apply:
            
            self._Standardisation()
            
        # PCA preprocess            
        if self.pcaPreproc.apply:
            
            self._PcaPreprocess()
        
        # RemoveOutliers is applied to the full dataset and affects all models
        if self.removeOutliers.apply:

            self._RemoveOutliers()

        # Any manual feature selection is applied to the original dataframe - i.e. affect all models the same
        if self.manualFeatureSelection.apply:

            self._ManualFeatureSelector()

        # The feature selection is applied to the original dataframe - i.e. affect all models the same
        if self.globalFeatureSelection.apply:

            self._VarianceSelector()

        # Set the subplot
        self._SetSubPlots()

        # Loop over the target features to model
        for self.targetN, self.targetFeature in enumerate(self.targetFeatures):

            if self.verbose:

                infoStr = '\n            Target feature: %s' %(self.targetFeature)

                print (infoStr)

            self._ExtractDataFrame()

            self._SetTargetFeatureSymbol()

            if self.targetFeatureSelection.apply:

                if self.targetFeatureSelection.univariateSelection.apply:

                    self._UnivariateSelector()

            # Covariate (X) Agglomeration
            if self.featureAgglomeration.apply:

                if self.featureAgglomeration.wardClustering.apply:

                    if self.featureAgglomeration.wardClustering.tuneWardClustering.apply:

                        n_clusters = self._TuneWardClustering()

                    else:

                        n_clusters = self.featureAgglomeration.wardClustering.n_clusters

                    self._WardClustering(n_clusters)
            # End of target feature related selection and clustering

            #Loop over the defined models
            for self.regrN, self.regrModel in enumerate(self.regressorModels):

                if  self.modelFeatureSelection.apply:

                    if self.modelFeatureSelection.permutationSelector.apply:

                        self._PermutationSelector()

                    elif self.modelFeatureSelection.RFE.apply:

                        if self.regrModel[0] in ['KnnRegr','MLP']:

                            self._PermutationSelector()

                        else:

                            self._RFESelector()

                if self.featureImportance.apply:

                    self._FeatureImportance(multCompAxs)

                if self.hyperParameterTuning.apply:

                    if self.hyperParameterTuning.exhaustiveTuning.apply:

                        self._ExhaustiveTuning()

                    elif self.hyperParameterTuning.randomTuning.apply:

                        self._RandomTuning()

                    # Reset the regressor with the optimized hyperparameter tuning

                    # Set the regressor models to apply
                    self._RegModelSelectSet()

                if self.verbose > 1:

                    # Report the regressor model settings (hyper parameters)
                    self._ReportRegModelParams()
                    
                # uschanged columns from the start as lists
                if type(self.columns) is list:
                    
                    self.finalFeatureLD[self.targetFeature][self.regrModel[0]] = self.columns
                
                else: # otherwise not

                    self.finalFeatureLD[self.targetFeature][self.regrModel[0]] = self.columns.tolist()

                if self.modelTests.apply:

                    if self.modelTests.trainTest.apply:

                        self._RegrModTrainTest(multCompAxs)

                    if self.modelTests.Kfold.apply:

                        self._RegrModKFold(multCompAxs)
                        
        if self.plot.rows.screenShow:

            plt.show()

        if self.plot.rows.savePng:

            if self.plot.rows.targetFeatures.apply:

                for regModel in self.paramD['regressionModels']:

                    if self.paramD['regressionModels'][regModel]['apply']:

                        self.columnFig[regModel].savefig(self.imageFPND[regModel]['alltargets'])

            if self.plot.rows.regressionModels.apply:

                for targetFeature in self.targetFeatures:

                    self.columnFig[targetFeature].savefig(self.imageFPND[targetFeature]['allmodels'])
                    
        for regModel in self.paramD['regressionModels']:

            if self.paramD['regressionModels'][regModel]['apply']:
        
                plt.close(fig=self.columnFig[regModel])
                
        for targetFeature in self.targetFeatures:
            
            if self.plot.rows.regressionModels.apply:

                plt.close(fig=self.columnFig[targetFeature])
        
        self._DumpJson()

def SetupProcesses(iniParams):
    '''Setup and loop processes

    :param rootpath: path to project root folder
    :type: lstr

    :param sourcedatafolder: folder name of original OSSL data (source folder)
    :type: lstr

    :param arrangeddatafolder: folder name of arranged OSSL data (destination folder)
    :type: lstr

    :param projFN: project filename (in destination folder)
    :type: str

    :param jsonpath: folder name
    :type: str

    '''

    dstRootFP, jsonFP = CheckMakeDocPaths(iniParams['rootpath'],
                                          iniParams['arrangeddatafolder'],
                                          iniParams['jsonfolder'],
                                          iniParams['sourcedatafolder'])

    if iniParams['createjsonparams']:

        CreateArrangeParamJson(jsonFP,iniParams['projFN'],'mlmodel')

    jsonProcessObjectD = ReadProjectFile(dstRootFP, iniParams['projFN'])
       
    #jsonProcessObjectL = jsonProcessObjectD['projectFiles']
    jsonProcessObjectL = [os.path.join(jsonFP,x.strip())  for x in jsonProcessObjectD['projectFiles'] if len(x) > 10 and x[0] != '#']
    
    # Get the target Feature Symbols
    targetFeatureSymbolsD = ReadAnyJson(iniParams['targetfeaturesymbols'])
    
    if not 'multiprojectcomparison' in jsonProcessObjectD or not jsonProcessObjectD['multiprojectcomparison']:
        
        multiProjectComparisonD = {'apply': False} 
    
    elif not os.path.exists(jsonProcessObjectD['multiprojectcomparison']):
        
        multiProjectComparisonD = {'apply': False} 
        
    else:    
    
        multiProjectComparisonD = ReadAnyJson(jsonProcessObjectD['multiprojectcomparison'])
        
    multCompFig = multCompAxs = False
    
    if multiProjectComparisonD['apply']:
        
        if (len(jsonProcessObjectL) < 2):
            
            exitStr = 'Exiting: multi comparison projects must have at least 2 project files\n    %s has only %s' %(iniParams['projFN'],(len(jsonProcessObjectL)))
            
            exitStr += '\n    Either add more projects or remove the "multiprojectcomparison" command'
            
            exit(exitStr)
                
        multCompImagesFPND, multCompJsonSummaryFPND = SetMultiCompDstFPNs(iniParams['rootpath'],iniParams['arrangeddatafolder'],
                                                                          multiProjectComparisonD,targetFeatureSymbolsD)
        
        multCompFig, multCompAxs, multCompPlotsColumns = SetMultCompPlots( multiProjectComparisonD,targetFeatureSymbolsD,len(jsonProcessObjectL) )

        multCompSummaryD = {}
        
        for targetFeature in multiProjectComparisonD['targetFeatures']:
            
            multCompSummaryD[targetFeature] = {}
            
        
    
        
    modelNr = 0    
    
    #Loop over all json files
    for jsonObj in jsonProcessObjectL:

        print ('    jsonObj:', jsonObj)

        paramD = ReadModelJson(jsonObj)
        
        if paramD['scattercorrection']['apply']:
                       
            if len(paramD['scattercorrection']['scaler']) == 0:
            
                paramD['scattercorrection']['apply'] = False
        
            elif len(paramD['scattercorrection']['scaler']) == 1:
                
                paramD['scattercorrection']['singles'] = \
                    paramD['scattercorrection']['scaler']
                    
                paramD['scattercorrection']['duals'] = []
                
            else:
                
                paramD['scattercorrection']['singles'] = []
      
                paramD['scattercorrection']['duals'] = \
                    paramD['scattercorrection']['scaler']
                           
        # Add the targetFeatureSymbols
        paramD['targetFeatureSymbols'] = targetFeatureSymbolsD['targetFeatureSymbols']
        
        # paramD['regressorSymbols'] =
       
        paramD['multcompplot'] = False
        
        if multiProjectComparisonD['apply']:
            
            paramD['multcompplot'] = True
            
            paramD['modelNr'] = modelNr
            
            paramD['multCompPlotsColumns'] = multCompPlotsColumns
            
            # Replace the list of targetFeatures in paramD
            paramD['targetFeatures'] = multiProjectComparisonD['targetFeatures']
            
            # Replace the applied regressors, but not the hyper parameter definitions    
            for regressor in paramD['regressionModels']:
                               
                paramD['regressionModels'][regressor]['apply'] = multiProjectComparisonD['regressionModels'][regressor]['apply'] 
                
            # Replace all the processing steps boolean apply
            '''processStepL = ['removeOutliers', 'manualFeatureSelection',
                            'globalFeatureSelection','targetFeatureSelection',
                            'modelFeatureSelection','featureAgglomeration',
                            'hyperParameterTuning']'''
            
            processStepL = ['standardisation','removeOutliers', 
                            'globalFeatureSelection','targetFeatureSelection',
                            'modelFeatureSelection','featureAgglomeration',
                            'hyperParameterTuning']
            
            for p in processStepL:
                
                paramD[p]['apply'] = multiProjectComparisonD[p]['apply']
                
            # Replace the feature importance reporting
            paramD['featureImportance'] = multiProjectComparisonD['featureImportance']
            
             
            # Replace the model test
            paramD['modelTests'] = multiProjectComparisonD['modelTests']                          
            
        # Get the target feature transform
        targetFeatureTransformD = ReadAnyJson(paramD['input']['targetfeaturetransforms'])
    
        # Add the targetFeatureTransforms
        paramD['targetFeatureTransform'] = targetFeatureTransformD['targetFeatureTransform']
        
        # Add the targetFeatureStandardisation
        paramD['targetFeatureStandardise'] = targetFeatureTransformD['targetFeatureStandardise']

        # Invoke the modeling
        mlModel = MachineLearningModel(paramD)

        mlModel.paramD = paramD

        # Add the raw paramD as a variable to mlModel
        mlModel.jsonparamsD = paramD

        # Set the regressor models to apply
        mlModel._RegModelSelectSet()
        
        mlModel._CheckParams(os.path.split(jsonObj)[1]);

        # run the modeling
        mlModel._PilotModeling(iniParams['rootpath'],iniParams['sourcedatafolder'],  dstRootFP, multCompFig, multCompAxs)
        
        
        if multiProjectComparisonD['apply']: 
            
            modelNrStr = '%s' %(modelNr)
            
            if modelNrStr in multiProjectComparisonD['trialid']:
                
                trialid = multiProjectComparisonD['trialid'][modelNrStr]
            
            else:
            
                trialid = 'trial_%s' %(modelNr)
            
            for targetFeature in mlModel.targetFeatures:
            
                multCompSummaryD[targetFeature][trialid] = mlModel.multCompSummaryD[targetFeature]
               
        modelNr += 1
    
    if multiProjectComparisonD['apply']: 
        
        print ('All models in project Done') 
        
        if multiProjectComparisonD['plot']['screenShow']:
        
            plt.show()
        
        #pp = pprint.PrettyPrinter(indent=2)
        #pp.pprint(multCompSummaryD)
        
        for targetFeature in multCompFig:
    
            for index in multCompFig[targetFeature]:
                
                jsonD = {targetFeature : multCompSummaryD[targetFeature]}
                
                DumpAnyJson(jsonD,multCompJsonSummaryFPND[targetFeature]) 
                
                if multiProjectComparisonD['plot']['savePng']: 
                             
                    multCompFig[targetFeature][index].savefig( multCompImagesFPND[targetFeature][index] )
        

if __name__ == '__main__':
    ''' If script is run as stand alone
    '''

    '''
    if len(sys.argv) != 2:

        sys.exit('Give the link to the json file to run the process as the only argument')

    #Get the json file
    rootJsonFPN = sys.argv[1]

    if not os.path.exists(jsonFPN):

        exitstr = 'json file not found: %s' %(rootJsonFPN)

    
    rootJsonFPN = "/Local/path/to/model_ossl.json"
    
    rootJsonFPN = "/Users/thomasgumbricht/docs-local/OSSL2/projects_model/model_ossl.json"
    
    rootJsonFPN = '/Users/thomasgumbricht/docs-local/OSSL2/projects_model/model_ossl_remove-outliers-comp.json'
    
    rootJsonFPN = '/Users/thomasgumbricht/docs-local/OSSL2/projects_model/model_ossl_glob-feat-select_comp.json'
    
    rootJsonFPN = '/Users/thomasgumbricht/docs-local/OSSL2/projects_model/model_ossl_tar-feat-select_comp.json'
    #rootJsonFPN = "/Users/thomasgumbricht/docs-local/OSSL2/model_ossl_yeojohnson.json"
    
    rootJsonFPN = '/Users/thomasgumbricht/docs-local/OSSL2/projects_model/model_ossl_tar-feat-agglom_comp.json'
    rootJsonFPN = '/Users/thomasgumbricht/docs-local/OSSL2/projects_model/model_ossl_tar-feat-agglom_comp.json'
    rootJsonFPN = '/Users/thomasgumbricht/docs-local/OSSL2/projects_model/model_ossl_tar-feat-agglom2_comp.json'
    #rootJsonFPN = '/Users/thomasgumbricht/docs-local/OSSL2/projects_model/model_theodor_error.json'
    
    rootJsonFPN = '/Users/thomasgumbricht/docs-local/OSSL2/projects_model/model_ossl_tar-feat-pretransform_comp.json'
    
    rootJsonFPN = '/Users/thomasgumbricht/docs-local/OSSL2/projects_model/model_ossl_AMS-sensors-wls.json'
    '''
    rootJsonFPN = '/Users/thomasgumbricht/docs-local/OSSL2/projects_model/hyper_model-OSSL-LUCAS-SE_600-870_54.json'

    rootJsonFPN = '/Users/thomasgumbricht/docs-local/OSSL2/projects_model/model_ossl_meancentring.json'
    
    iniParams = ReadAnyJson(rootJsonFPN)
            
if type( iniParams['projFN']) is list: 
          
        for proj in iniParams['projFN']:
            
            projParams = deepcopy(iniParams)
            
            projParams['projFN'] = proj
            
            SetupProcesses(projParams)
           
else:
        
    SetupProcesses(iniParams)
