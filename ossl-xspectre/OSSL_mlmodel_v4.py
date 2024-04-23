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
from numpy import integer


'''
TARGET FEATURE QUANTILE
TARGEt FEATURE BOX-COX MM
TUNEWARDCLUSTERING WHY USE BAYESIANRIDGE???

'''
# Standard library imports

import os

import json

import datetime

from copy import deepcopy

from math import sqrt

from time import sleep

import pprint

import csv

# Third party imports
import tempfile

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from numbers import Integral, Real

from scipy.stats import randint as sp_randint

from scipy.stats import boxcox

import pickle

from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, normalize
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression

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
#from sklearn.feature_selection import SelectFromModel

from sklearn.inspection import permutation_importance
from sklearn.metrics._regression import mean_absolute_error,\
    mean_absolute_percentage_error, median_absolute_error
    
from sklearn.inspection import DecisionBoundaryDisplay
    
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal import savgol_filter

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

    #paramD['targetFeatureSymbols'] = {'caco3_usda.a54_w.pct':{'color': 'orange', 'size':50}}

    paramD['derivatives'] = {'apply':False, 'join':False}
    
    paramD['scatterCorrection'] = {}
    
    paramD['scatterCorrection']['comment'] = "scattercorr can apply the following: norm-1 (l1), norm-2 (l2) norm-max (max), snv (snv) or msc (msc). If two are gicen they will be done in series using output from the first as input to the second"

    paramD['scatterCorrection']['apply'] = False
    
    paramD['scatterCorrection']['scaler'] = []
    
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

    paramD['generalFeatureSelection'] = {}

    paramD['generalFeatureSelection']['comment'] ="removes spectra with variance below given thresholds - globally applied as preprocess",

    paramD['generalFeatureSelection']['apply'] = False

    paramD['generalFeatureSelection']['varianceThreshold'] = {'threshold': 0.025}

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

def SetcolorRamp(n, colorRamp):
        ''' Slice predefined colormap to discrete colors for each band
        '''

        # Set colormap to use for plotting
        cmap = plt.get_cmap(colorRamp)

        # Segmenting colormap to the number of bands
        slicedCM = cmap(np.linspace(0, 1, n))
        
        return (slicedCM)
   
def SetFigSize(xadd, yadd, sizeX, sizeY, subFigSizeX, subFigSizeY, cols, rows):
    '''  Set the figure size
    '''

    if sizeX == 0:

        figSizeX = subFigSizeX * rows + xadd

    else:

        figSizeX = subFigSizeX


    if sizeY == 0:

        figSizeY = subFigSizeY * cols + yadd

    else:

        figSizeY = subFigSizeY 
        
    return (figSizeX, figSizeY)

def GetPlotStyle(plotLayout):
    
    if plotLayout.linewidth: # linewidth == 0, no lines
        
        plotStyle = plotLayout.linestyle
    
    else:
        
        plotStyle = ''
        
    if plotLayout.pointsize:
        
        plotStyle += plotLayout.pointstyle
        
    return plotStyle

def PlotFilterExtract(plotLayout, filterTxt, originalDF, filterDF, plotFPN):
    """
    """
    from math import ceil
    
    # Get the plot layout arguments
    maxSpectra = plotLayout.maxSpectra
            
    subplotTitles = plotLayout.scatterCorrection.subplotTitles
    
    # Get the bands to plot
    plotskipStep = ceil( (len(originalDF.index)-1)/maxSpectra )
    
    # Create empty dict to hold the data
    subplotsD = {}
    
    subplotsD['original'] = {}
    
    subplotsD['filtered'] = {}
    
    subplotsD['original'] = {'label': 'Original data',
                                      'DF' : originalDF}
    
    subplotsD['filtered'] = { 'label': 'Filtered/extracted data',
                                      'DF' : filterDF}
        
    #fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(self.spectraPlot.subfigs.figSize.x, self.spectraPlot.subfigs.figSize.y), sharex=True  )
    filterfig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True, sharey=True )

    n = int(len(originalDF.index)/plotskipStep)+1
        
    # With n bands known, create the colorRamp
    slicedCM = SetcolorRamp(n, plotLayout.colorRamp)
    
    # Extract the columns bands) as floating wavelenhts   
    #columnsX = [float(item) for item in self.spectraDF.columns]
        
    origx = [float(i) for i in originalDF.columns]
    
    filterx =  [float(i) for i in filterDF.columns]
        
    # Get the plot style
    plotStyle =  GetPlotStyle(plotLayout)
                         
    for c, key in enumerate(subplotsD):
        
        if c == 1:
                
            ax[c].set(xlabel='wavelength')
            
            if plotLayout.filterExtraction.annotate.filtered:
                
                if plotLayout.filterExtraction.annotate.filtered == 'auto':
                 
                    txtStr = 'Filtered spectra\n  %s\n  %s total bands\n  showing every %s band' %(filterTxt, len(filterx),plotskipStep)
                    
                else:
                    
                    txtStr = plotLayout.filterExtraction.annotate.filtered
            
                ax[c].annotate(txtStr,
                           (plotLayout.filterExtraction.annotate.x,
                            plotLayout.filterExtraction.annotate.y),
                           xycoords = 'axes fraction' )
        else:
            
            if plotLayout.filterExtraction.annotate.original:
                
                if plotLayout.filterExtraction.annotate.original == 'auto':
                   
                    txtStr = 'Original spectra\n  %s total bands\n  showing every %s band' %(len(origx),plotskipStep) 
                
                else:
                    
                    txtStr = plotLayout.filterExtraction.annotate.original
  
                ax[c].annotate(txtStr,
                           (plotLayout.filterExtraction.annotate.x,
                            plotLayout.filterExtraction.annotate.y),
                           xycoords = 'axes fraction' )
                      
        ax[c].set(ylabel=plotLayout.filterExtraction.ylabels[c])
                                        
        # Loop over the spectra
        i = -1
        
        n = 0
            
        for index, row in subplotsD[key]['DF'].iterrows():
                
            i += 1
            
            if i % plotskipStep == 0:
                
                if c == 0:
                    
                    ax[c].plot(origx, row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                else:
                    
                    ax[c].plot(filterx, row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 
                    
                n += 1
                
    # Set supTitle
    if plotLayout.filterExtraction.supTitle:
        
        if plotLayout.filterExtraction.supTitle == 'auto':
            
            filterfig.suptitle('Scatter Correction')
        
        else:
    
            filterfig.suptitle(plotLayout.filterExtraction.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        filterfig.tight_layout() 
        
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
    
        filterfig.savefig(plotFPN)   # save the figure to file
           
    plt.close(filterfig)

                          
def PlotScatterCorr(plotLayout, plotFPN, corrTxtL, columns,
         trainOriginalDF, testOriginalDF,  
                    trainCorr1DF, testCorr1DF, trainCorr2DF=None, testCorr2DF=None):
    """
    """
    from math import ceil
    
    # Get the plot layout arguments
    maxSpectra = plotLayout.maxSpectra
            
    subplotTitles = plotLayout.scatterCorrection.subplotTitles
    
    # Get the bands to plot
    plotskipStep = ceil( (len(trainOriginalDF.index)-1)/maxSpectra )
    
    ttratio = plotskipStep / ceil( (len(testOriginalDF.index)-1)/maxSpectra)
    
    annotateStrD = {0:'', 1:'', 2:'',}
    
    if plotLayout.scatterCorrection.annotate.input:          
        if plotLayout.scatterCorrection.annotate.input == 'auto':
            annotateStrD[0] = 'Input spectra\n  showing every %s band' %(plotskipStep)
        else:
            annotateStrD[0] = plotLayout.scatterCorrection.annotate.input
            
    if plotLayout.scatterCorrection.annotate.firstcorrect:          
        if plotLayout.scatterCorrection.annotate.firstcorrect == 'auto':
            annotateStrD[1] = 'After %s correction\n  showing every %s band' %(corrTxtL[0], plotskipStep)
        else:
            annotateStrD[1] = plotLayout.scatterCorrection.annotate.firstcorrect
    
    if len(corrTxtL) > 1 and plotLayout.scatterCorrection.annotate.secondcorrect:          
        if plotLayout.scatterCorrection.annotate.secondcorrect == 'auto':
            annotateStrD[2] = 'After %s correction\n  showing every %s band' %(corrTxtL[1], plotskipStep)
        else:
            annotateStrD[2] = plotLayout.scatterCorrection.annotate.secondcorrect        
            
    # Create empty dict to hold the data
    subplotsD = {}
    
    subplotsD['original'] = {}
        
    subplotsD['original']['train'] = {'label': 'Training data (original)',
                                      'DF' : trainOriginalDF}
    
    subplotsD['original']['test'] = { 'label': 'Test data (original)',
                                      'DF' : testOriginalDF}
        
        
    
    
    if len(corrTxtL) == 1:
        
        rmax = 1

        #fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(self.spectraPlot.subfigs.figSize.x, self.spectraPlot.subfigs.figSize.y), sharex=True  )
        scatplotfig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey='row' )
        
        subplotsD['corrfinal'] = {}
        
        subplotsD['corrfinal']['train'] = { 'label': 'Scatter correction: %s' %(corrTxtL[0]),
                                       'DF' : trainCorr1DF}
    
        subplotsD['corrfinal']['test'] = { 'label': 'Scatter correction: %s' %(corrTxtL[0]),
                                    'DF' : testCorr1DF}

    else:

        rmax = 2
        
        #fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(self.spectraPlot.subfigs.figSize.x, self.spectraPlot.subfigs.figSize.y), sharex=True  )
        scatplotfig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 8), sharex=True, sharey='row' )
        
        # Intermed must be declared before final
        subplotsD['corrintermed'] = {}
        
        subplotsD['corrfinal'] = {}
                
        subplotsD['corrintermed']['train'] = { 'column': 0, 'row':1,
                                    'label': 'Scatter correction: %s' %(corrTxtL[0]),
                                    'DF' : trainCorr2DF}
    
        subplotsD['corrintermed']['test'] = { 'column': 1, 'row':1,
                                    'label': 'Scatter correction: %s' %(corrTxtL[0]),
                                    'DF' : testCorr2DF}
        
        subplotsD['corrfinal']['train'] = { 'column': 0, 'row':2,
                                    'label': 'Scatter correction: %s' %(corrTxtL[1]),
                                    'DF' : trainCorr1DF}
    
        subplotsD['corrfinal']['test'] = { 'column': 1, 'row':2,
                                    'label': 'scatter correct. %s' %(corrTxtL[1]),
                                    'DF' : testCorr1DF}
   
    n = int(len(trainOriginalDF.index)/plotskipStep)+1
        
    # With n bands known, create the colorRamp
    slicedCM = SetcolorRamp(n, plotLayout.colorRamp)
    
    x_spectra_integers = [int(i) for i in columns]
    
    # Get the plot style
    plotStyle =  GetPlotStyle(plotLayout)
    
    for r, key in enumerate(subplotsD):
        
        if r == rmax:
                
                for c in range(len(subplotsD[key])):
                
                    ax[r][c].set(xlabel='wavelength')
        
        for c, subplotkey in enumerate(subplotsD[key]):
            
            ax[r][c].set( title=subplotsD[key][subplotkey]['label'])
            
            ax[r][c].annotate(annotateStrD[r],
                           (plotLayout.scatterCorrection.annotate.x,
                            plotLayout.scatterCorrection.annotate.y),
                           xycoords = 'axes fraction' )
            
            if c == 0:
                  
                ax[r][c].set(ylabel=plotLayout.scatterCorrection.ylabels[r])
                                        
            # Loop over the spectra
            i = -1
        
            n = 0
            
            for index, row in subplotsD[key][subplotkey]['DF'].iterrows():
                
                i += 1
                
                if i % plotskipStep == 0:
                                         
                    if c == 0:
                        #ax[r][c].plot(x_spectra_integers, row, color=slicedCM[n])
                        
                        ax[r][c].plot(x_spectra_integers, row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                        
                    else:
                        
                        m = ceil(n*ttratio)
                        
                        #ax[r][c].plot(x_spectra_integers, row, color=slicedCM[m])
                        
                        ax[r][c].plot(x_spectra_integers, row, plotStyle, color=slicedCM[m], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                                
                    n += 1
                
    # Set supTitle
    if plotLayout.scatterCorrection.supTitle:
        
        if plotLayout.scatterCorrection.supTitle == 'auto':
            
            scatplotfig.suptitle('Scatter Correction')
        
        else:
    
            scatplotfig.suptitle(plotLayout.scatterCorrection.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        scatplotfig.tight_layout()
            
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
    
        scatplotfig.savefig(plotFPN)   # save the figure to file
    
        #infostr = 'Plots of scatter correction saved as:\n    %s' %(plotFPN)
        
        #print(infostr)
         
    plt.close(scatplotfig)

def ScatterCorrectDF(trainDF, testDF, scattercorrection, columns):
    """ Scatter correction for spectral signals

        :returns: scatter corrected spectra
        :rtype: pandas dataframe
    """
    
    normD = {}
    
    normD['l1'] = { 'norm':'l1', 'label':'L1norm'}
    normD['l2'] = {'norm':'l2', 'label':'L2norm'}
    normD['max'] = {'norm':'max', 'label':'maxnorm'}
    normD['snv'] = {'norm':'max', 'label':'SNV'}
    normD['msc'] = {'norm':'max', 'label':'MSC'}
    
    M1 = [None, None]
    
    firstCorrTrainDf = None
                
    firstCorrTestDf = None
        
    scattCorrTxt = 'None'
    
    corrTxtL = []
    
    for singleScattCorr in scattercorrection.singles:
        
        if not singleScattCorr in normD:
            
            exitStr = 'EXITING - unrecognized scatter correction method: %s' %(singleScattCorr)
            
            exit(exitStr) 
            
        scattCorrTxt = normD[singleScattCorr]['label']
        
        corrTxtL.append(scattCorrTxt)
        
        #trainDFD[scatcorr] = {'label': normD[scatcorr]['label']}
        #testDFD[scatcorr] = {'label': normD[scatcorr]['label']}
        
        if singleScattCorr in ['l1','l2','max']:
            
            X1, N1 = normalize(trainDF, norm=singleScattCorr, return_norm=True) 
            
            X2, N2 = normalize(testDF, norm=singleScattCorr, return_norm=True) 

            trainDF = pd.DataFrame(data=X1, columns=columns)
            testDF = pd.DataFrame(data=X2, columns=columns)
            
        elif singleScattCorr == 'snv':
            
            X1 = np.array(trainDF[columns])
            
            X1 = snv(X1)
            
            X2 = np.array(testDF[columns])
            
            X2 = snv(X2)
            
            trainDF = pd.DataFrame(data=X1, columns=columns)
            testDF = pd.DataFrame(data=X2, columns=columns)

        elif singleScattCorr == 'msc':

            X1 = np.array(trainDF[columns])
            
            X1, M1[0] = msc(X1)
            
            X2 = np.array(testDF[columns])
            
            X2, M2 = msc(X2,M1[0])

            trainDF = pd.DataFrame(data=X1, columns=columns)
            testDF = pd.DataFrame(data=X2, columns=columns)
            
        else:
            
            exitStr = 'EXITING - unrecognized scatter correction method: %s' %(singleScattCorr)
            
            exit(exitStr)
            
    for s, dualScattCorr in enumerate(scattercorrection.duals):
        
        if not dualScattCorr in normD:
            
            exitStr = 'EXITING - unrecognized scatter correction method: %s' %(dualScattCorr)
            
            exit(exitStr)
        
        if s == 0:
            
            scattCorrTxt = normD[dualScattCorr]['label']
        
        else:
            
            scattCorrTxt += '+%s' %(normD[dualScattCorr]['label'])
   
        corrTxtL.append(scattCorrTxt)
        
        dualTrainDF = deepcopy(trainDF)
        dualTestDF = deepcopy(testDF)
                    
        print ('scatcorr',dualScattCorr)

        if dualScattCorr in ['l1','l2','max']:
            
            X1 = np.array(dualTrainDF[columns])
            
            X1 = normalize(X1, norm=dualScattCorr ) 
                        
            X2 = np.array(dualTestDF[columns])
            
            X2 = normalize(X2, norm=dualScattCorr ) 
          
        elif dualScattCorr  == 'snv':
            
            X1 = np.array(dualTrainDF[columns])
            
            X1 = snv(X1)
            
            X2 = np.array(dualTestDF[columns])
            
            X2 = snv(X2)
                  
        elif dualScattCorr == 'msc':
            
            X1 = np.array(dualTrainDF[columns])
            
            X1, M1[s] = msc(X1)
            
            X2 = np.array(dualTestDF[columns])
            
            X2, M2 = msc(X2,M1[s])
            
        else:
            
            exitStr = 'EXITING - unrecognized scatter correction method: %s' %(dualScattCorr)
            
            exit(exitStr)
            
        dualTrainDF = pd.DataFrame(data=X1, columns=columns)
            
        dualTestDF = pd.DataFrame(data=X2, columns=columns)
            
        if s == 0:
    
            firstCorrTrainDf = deepcopy(dualTrainDF)
            
            firstCorrTestDf = deepcopy(dualTestDF)
                
        trainDF= dualTrainDF
        
        testDF = dualTestDF
            
    return scattCorrTxt, corrTxtL, trainDF, testDF, firstCorrTrainDf, firstCorrTestDf, M1
                      
def ScatterCorrection(trainDF, testDF, scattercorrection, plotLayout, plotFPN):
    """ Scatter correction for spectral signals

        :returns: organised spectral derivates
        :rtype: pandas dataframe
    """
    
    columns = [item for item in trainDF]
        
    origTrainDF = deepcopy(trainDF)
    
    origTestDF = deepcopy(testDF)
    
    scattCorrTxt, corrTxtL, trainDF, testDF, firstCorrTrainDF, firstCorrTestDF, scatCorrMeanSpectraL = ScatterCorrectDF(trainDF, testDF, scattercorrection, columns)
    
    #NoN can develop and must be removed

    PlotScatterCorr(plotLayout, plotFPN, corrTxtL, columns, origTrainDF, origTestDF, trainDF, testDF, firstCorrTrainDF, firstCorrTestDF)

    return scattCorrTxt, trainDF, testDF,  scatCorrMeanSpectraL

def PlotDerivatives(X_train, X_test, X_train_derivative, X_test_derivative, 
                    columns, dColumns, plotLayout, plotFPN):
    """
    """
    from math import ceil
    
    # Get the plot layout arguments
    maxSpectra = plotLayout.maxSpectra
            
    subplotTitles = plotLayout.derivative.subplotTitles
    
    # Get the bands to plot
    plotskipStep = ceil( (len(X_train.index)-1)/maxSpectra )
    
    # ttration = trian-test ratio - only for adjusting colorramp
    ttratio = plotskipStep / ceil( (len(X_test.index)-1)/maxSpectra)
    
    # Create empty dict to hold the data
    subplotsD = {}
    
    subplotsD['input'] = {}
    
    subplotsD['derivative'] = {}
    
    subplotsD['input']['train'] = {'label': 'Training data (input)',
                                      'DF' : X_train}
    
    subplotsD['input']['test'] = { 'label': 'Test data (input)',
                                      'DF' : X_test}
    
    subplotsD['derivative']['train'] = {'label': 'Derivative',
                                      'DF' : X_train_derivative}
    
    subplotsD['derivative']['test'] = { 'label': 'Derivatives',
                                      'DF' : X_test_derivative}
  
    #fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(self.spectraPlot.subfigs.figSize.x, self.spectraPlot.subfigs.figSize.y), sharex=True  )
    derivativeplotfig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey='row' )
        
    n = int(len(X_train.index)/plotskipStep)+1
        
    # With n bands known, create the colorRamp
    slicedCM = SetcolorRamp(n, plotLayout.colorRamp)
    
    annotateStrD = {0:'', 1:'', 2:'',}
    
    if plotLayout.derivative.annotate.input:          
        if plotLayout.derivative.annotate.input == 'auto':
            annotateStrD[0] = 'Input spectra\n  showing every %s band' %(plotskipStep)
        else:
            annotateStrD[0] = plotLayout.derivative.annotate.input
            
    if plotLayout.derivative.annotate.derivative:          
        if plotLayout.derivative.annotate.derivative == 'auto':
            annotateStrD[1] = 'Derivatives\n  showing every %s band' %(plotskipStep)
        else:
            annotateStrD[1] = plotLayout.derivative.annotate.derivative
    
    # Get the plot style
    plotStyle =  GetPlotStyle(plotLayout)
    
    xD = {}
    
    xD[0] = list(columns.values())
    
    xD[1] = list(dColumns.values())
    
    for r, key in enumerate(subplotsD):
        
        if r == 1:
                
                for c in range(len(subplotsD[key])):
                
                    ax[r][c].set(xlabel='wavelength')
        
        for c, subplotkey in enumerate(subplotsD[key]):
            
            ax[r][c].annotate(annotateStrD[r],
                           (plotLayout.scatterCorrection.annotate.x,
                            plotLayout.scatterCorrection.annotate.y),
                           xycoords = 'axes fraction' )
            
            ax[r][c].set( title=subplotsD[key][subplotkey]['label'])
            
            if c == 0:
                  
                ax[r][c].set(ylabel=plotLayout.derivative.ylabels[r])
                                        
            # Loop over the spectra
            i = -1
        
            n = 0
            
            for index, row in subplotsD[key][subplotkey]['DF'].iterrows():
                
                i += 1
                
                if i % plotskipStep == 0:
                                         
                    if c == 0:

                        ax[r][c].plot(xD[r], row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 
  
                    else:
                        
                        m = ceil(n*ttratio)
                                               
                        ax[r][c].plot(xD[r], row, plotStyle, color=slicedCM[m], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 
            
                    n += 1
                
    # Set supTitle
    if plotLayout.derivative.supTitle:
        
        if plotLayout.derivative.supTitle == 'auto':
            
            derivativeplotfig.suptitle('Derivative')
        
        else:
    
            derivativeplotfig.suptitle(plotLayout.derivative.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        derivativeplotfig.tight_layout()
                        
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
    
        derivativeplotfig.savefig(plotFPN)   # save the figure to file
    
        #infostr = 'Plots of standardisation saved as:\n    %s' %(plotFPN)
        
        #print(infostr)
         
    plt.close(derivativeplotfig)
    
def PlotPCA(plotLayout, plotFPN, pcaTxt, columnsD,
         trainInputDF, testInputDF, trainPCADF, testPCADF):
    """ Combine with standatdisation and derivation etc
    """
    from math import ceil
    
    # Get the plot layout arguments
    maxSpectra = plotLayout.maxSpectra
            
    subplotTitles = plotLayout.pca.subplotTitles
    
    # Get the bands to plot
    plotskipStep = ceil( (len(trainInputDF.index)-1)/maxSpectra )
    
    # ttration = trian-test ratio - only for adjusting colorramp
    ttratio = plotskipStep / ceil( (len(testInputDF.index)-1)/maxSpectra)
    
    # Create empty dict to hold the data
    subplotsD = {}
    
    subplotsD['original'] = {}
    
    subplotsD['standardised'] = {}
    
    subplotsD['original']['train'] = {'label': 'Training data (input)',
                                      'DF' : trainInputDF}
    
    subplotsD['original']['test'] = { 'label': 'Test data (input)',
                                      'DF' : testInputDF}
    
    subplotsD['standardised']['train'] = {'label': 'Principal components',
                                      'DF' : trainPCADF}
    
    subplotsD['standardised']['test'] = { 'label': 'Principal components',
                                      'DF' : testPCADF}
  
    #fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(self.spectraPlot.subfigs.figSize.x, self.spectraPlot.subfigs.figSize.y), sharex=True  )
    pcaplotfig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex='row', sharey='row' )
        
    n = int(len(trainInputDF.index)/plotskipStep)+1
        
    # With n bands known, create the colorRamp
    slicedCM = SetcolorRamp(n, plotLayout.colorRamp)
    
    
    
    annotateStrD = {0:'', 1:'', 2:'',}
    
    if plotLayout.pca.annotate.input:          
        if plotLayout.pca.annotate.input == 'auto':
            annotateStrD[0] = 'Input spectra\n  showing every %s band' %(plotskipStep)
        else:
            annotateStrD[0] = plotLayout.pca.annotate.input
            
    if plotLayout.pca.annotate.output:          
        if plotLayout.pca.annotate.output == 'auto':
            annotateStrD[1] = 'Eigen vectors\n  showing every %s band' %(plotskipStep)
        else:
            annotateStrD[1] = plotLayout.pca.annotate.output
    
    # Get the plot style
    plotStyle =  GetPlotStyle(plotLayout)

    for r, key in enumerate(subplotsD):
        
        if r == 0:
                
                for c in range(len(subplotsD[key])):
                
                    ax[r][c].set(xlabel='wavelength')
                    
                x_spectra_integers = list(columnsD.values())
                    
        if r == 1:
                
            for c in range(len(subplotsD[key])):
                
                ax[r][c].set(xlabel='component')
                
                x_spectra_integers = np.arange(0,len(trainPCADF.columns))
        
        for c, subplotkey in enumerate(subplotsD[key]):
            
            ax[r][c].annotate(annotateStrD[r],
                           (plotLayout.scatterCorrection.annotate.x,
                            plotLayout.scatterCorrection.annotate.y),
                           xycoords = 'axes fraction' )
            
            ax[r][c].set( title=subplotsD[key][subplotkey]['label'])
            
            if c == 0:
                  
                ax[r][c].set(ylabel=plotLayout.pca.ylabels[r])
                                        
            # Loop over the spectra
            i = -1
        
            n = 0
            
            for index, row in subplotsD[key][subplotkey]['DF'].iterrows():
                
                i += 1
                
                if i % plotskipStep == 0:
                                         
                    if c == 0:
                        
                        ax[r][c].plot(x_spectra_integers, row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 
   
                    else:
                        
                        m = ceil(n*ttratio)
                                                
                        ax[r][c].plot(x_spectra_integers, row, plotStyle, color=slicedCM[m], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 
           
                    n += 1
                
    # Set supTitle
    if plotLayout.pca.supTitle:
        
        if plotLayout.pca.supTitle == 'auto':
            
            pcaplotfig.suptitle('Principal Component Analysis (PCA)')
        
        else:
    
            pcaplotfig.suptitle(plotLayout.pca.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        pcaplotfig.tight_layout()
                
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
    
        pcaplotfig.savefig(plotFPN)  
         
    plt.close(pcaplotfig)
    
def PlotStandardisation(plotLayout, plotFPN, standardTxt, columns,
         trainOriginalDF, testOriginalDF, trainStandarisedDF, testStandarisedDF):
    """
    """
    from math import ceil
    
    # Get the plot layout arguments
    maxSpectra = plotLayout.maxSpectra
            
    subplotTitles = plotLayout.standardisation.subplotTitles
    
    # Get the bands to plot
    plotskipStep = ceil( (len(trainOriginalDF.index)-1)/maxSpectra )
    
    # ttration = trian-test ratio - only for adjusting colorramp
    ttratio = plotskipStep / ceil( (len(testOriginalDF.index)-1)/maxSpectra)
    
    # Create empty dict to hold the data
    subplotsD = {}
    
    subplotsD['original'] = {}
    
    subplotsD['standardised'] = {}
    
    subplotsD['original']['train'] = {'label': 'Training data (input)',
                                      'DF' : trainOriginalDF}
    
    subplotsD['original']['test'] = { 'label': 'Test data (input)',
                                      'DF' : testOriginalDF}
    
    subplotsD['standardised']['train'] = {'label': 'Standardisation: %s' %(standardTxt),
                                      'DF' : trainStandarisedDF}
    
    subplotsD['standardised']['test'] = { 'label': 'Standardisation: %s' %(standardTxt),
                                      'DF' : testStandarisedDF}
  
    #fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(self.spectraPlot.subfigs.figSize.x, self.spectraPlot.subfigs.figSize.y), sharex=True  )
    standardplotfig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey='row' )
        
    n = int(len(trainOriginalDF.index)/plotskipStep)+1
        
    # With n bands known, create the colorRamp
    slicedCM = SetcolorRamp(n, plotLayout.colorRamp)
    
    x_spectra_integers = [int(i) for i in columns]
    
    annotateStrD = {0:'', 1:'', 2:'',}
    
    if plotLayout.standardisation.annotate.input:          
        if plotLayout.standardisation.annotate.input == 'auto':
            annotateStrD[0] = 'Input spectra\n  showing every %s band' %(plotskipStep)
        else:
            annotateStrD[0] = plotLayout.standardisation.annotate.input
            
    if plotLayout.standardisation.annotate.standard:          
        if plotLayout.standardisation.annotate.standard == 'auto':
            annotateStrD[1] = 'After %s\n  showing every %s band' %(standardTxt, plotskipStep)
        else:
            annotateStrD[1] = plotLayout.standardisation.annotate.standard
    
    # Get the plot style
    plotStyle =  GetPlotStyle(plotLayout)

    for r, key in enumerate(subplotsD):
        
        if r == 1:
                
                for c in range(len(subplotsD[key])):
                
                    ax[r][c].set(xlabel='wavelength')
        
        for c, subplotkey in enumerate(subplotsD[key]):
            
            ax[r][c].annotate(annotateStrD[r],
                           (plotLayout.scatterCorrection.annotate.x,
                            plotLayout.scatterCorrection.annotate.y),
                           xycoords = 'axes fraction' )
            
            ax[r][c].set( title=subplotsD[key][subplotkey]['label'])
            
            if c == 0:
                  
                ax[r][c].set(ylabel=plotLayout.scatterCorrection.ylabels[r])
                                        
            # Loop over the spectra
            i = -1
        
            n = 0
            
            for index, row in subplotsD[key][subplotkey]['DF'].iterrows():
                
                i += 1
                
                if i % plotskipStep == 0:
                                         
                    if c == 0:
                        #ax[r][c].plot(x_spectra_integers, row, color=slicedCM[n])
                        
                        ax[r][c].plot(x_spectra_integers, row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                        
                    else:
                        
                        m = ceil(n*ttratio)
                        
                        #ax[r][c].plot(x_spectra_integers, row, color=slicedCM[m])
                        
                        ax[r][c].plot(x_spectra_integers, row, plotStyle, color=slicedCM[m], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                                
                    n += 1
                
    # Set supTitle
    if plotLayout.standardisation.supTitle:
        
        if plotLayout.standardisation.supTitle == 'auto':
            
            standardplotfig.suptitle('Standardisation')
        
        else:
    
            standardplotfig.suptitle(plotLayout.standardisation.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        standardplotfig.tight_layout()
                
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
    
        standardplotfig.savefig(plotFPN)   # save the figure to file
    
        #infostr = 'Plots of standardisation saved as:\n    %s' %(plotFPN)
        
        #print(infostr)
         
    plt.close(standardplotfig)
    
def Standardisation(X_train, X_test, standardisation, plotLayout, plotFPN):
        """
        """
        
        scalertxt = 'None'
        
        origTrain = deepcopy(X_train)
    
        origTest = deepcopy(X_test)
        
        columns = [item for item in X_train.columns]
        
        # extract the covariate columns as X
        X1 = X_train[columns]
        
        X2 = X_test[columns]
        
        scaler = StandardScaler().fit(X1)
                
        arrayLength = scaler.var_.shape[0]
        
        if standardisation.paretoscaling:
            #  No meancentring, scales each variable by the square root of the standard deviation

            # remove meancentring            
            scaler.mean_ = np.zeros(arrayLength)
                      
            # set var_ to its own square root
            scaler.var_ = np.sqrt(scaler.var_)
            
            # set scaling to the sqrt of the std
            scaler.scale_ = np.sqrt(scaler.var_)
                        
            X1A = scaler.transform(X1)  
            
            X2A = scaler.transform(X2) 

            scalertxt = 'Pareto'          
                
        elif standardisation.poissonscaling:
            # No meancentring, scales each variable by the square root of the mean of the variable
            
            scaler = StandardScaler(with_mean=False).fit(X1)
            
            # Set var_ to mean_
            scaler.var_ = scaler.mean_
            
            # set scaler to sqrt of mean_ (var_)
            scaler.scale_ = np.sqrt(scaler.var_)
            
            X1A = scaler.transform(X1) 
            
            X2A = scaler.transform(X2) 
            
            scalertxt = 'Poisson' 
            
        elif standardisation.meancentring:
            
            if standardisation.unitscaling:
                
                # This is a classical autoscaling or z-score normalisation
                X1A = StandardScaler().fit_transform(X1)
                
                X2A = StandardScaler().fit_transform(X2)
                
                scalertxt = 'Z-score' 
                   
            else:
                
                # This is meancentring
                X1A = StandardScaler(with_std=False).fit_transform(X1)
                
                X2A = StandardScaler(with_std=False).fit_transform(X2)
                
                scalertxt = 'meancentring' 
        
        elif standardisation.unitscaling:
            
            X1A = StandardScaler(with_mean=False).fit_transform(X1)
                
            X2A = StandardScaler(with_mean=False).fit_transform(X2)
            
            scalertxt = 'deviation'
        
        else:
            
            standardisation.apply = False
            
            exit('EXITING - standardisation.apply is set to true but no method defined\n either set standardisation.apply to false or pick a method')
            
        if standardisation.apply:
            # Reset the train and test dataframes                
            X_train = pd.DataFrame(data=X1A, columns=columns)
            X_test = pd.DataFrame(data=X2A, columns=columns)
                  
            PlotStandardisation(plotLayout, plotFPN, scalertxt, columns,
                                origTrain, origTest, X_train, X_test)
            
     
        return X_train, X_test, scalertxt, scaler.mean_, scaler.scale_
     
def Derivatives(X_train, X_test, joinDerivative, Xcolumns, plotLayout, plotFPN):
    '''
    '''
        
    columnsStr = list(Xcolumns.keys())
    
    columnsNum = list(Xcolumns.values())
    
    
    # Get the derivatives
    X_train_derivative = X_train.diff(axis=1, periods=1)

    # Drop the first column as it will have only NaN
    X_train_derivative = X_train_derivative.drop(columnsStr[0], axis=1)

    # Create the derivative columns
    derivativeColumnsNum = [ (columnsNum[i-1]+columnsNum[i])/2 for i in range(len(columnsNum)) if i > 0]
    
    # Check the numeric format of derivativeColumnsNum:
    allIntegers = True
    
    for item in derivativeColumnsNum:

        if item % int(item) != 0:
            
            allIntegers = False
            
            break
       
    if allIntegers:
        
        derivativeColumnsNum = [int(item) for item in derivativeColumnsNum]
        
        derivativeColumnsStr = ['d%s' % i for i in derivativeColumnsNum]
    
    else:
        
        derivativeColumnsStr = ['d%.1f' % i for i in derivativeColumnsNum]
             
    dColumns = dict(zip(derivativeColumnsStr,derivativeColumnsNum) )

    # Replace the columns
    X_train_derivative.columns = derivativeColumnsStr
    
    # Repeat with test data
    # Get the derivatives
    X_test_derivative = X_test.diff(axis=1, periods=1)

    # Drop the first column as it will have only NaN
    X_test_derivative = X_test_derivative.drop(columnsStr[0], axis=1)
    
    # Replace the columns
    X_test_derivative.columns = derivativeColumnsStr
    
    PlotDerivatives(X_train, X_test, X_train_derivative, X_test_derivative, Xcolumns, dColumns, plotLayout, plotFPN )
    
    if joinDerivative:

        X_train_frames = [X_train, X_train_derivative]

        X_train = pd.concat(X_train_frames, axis=1)
        
        X_test_frames = [X_test, X_test_derivative]

        X_test = pd.concat(X_test_frames, axis=1)
        
        columns = {**Xcolumns, **dColumns }
        
    else:

        X_train = X_train_derivative
        
        X_test = X_test_derivative
        
        columns = dColumns
        
    return X_train, X_test, columns
       
def SetMultiCompDstFPNs(rootPath, arrangeDataPath, multiProjectComparisonD):
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
                   
    indexL = ['coefficientImportance','permutationImportance','treeBasedImportance','trainTest','Kfold']
    
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

def SetMultCompPlots(multiProjectComparisonD, targetFeatureSymbolsD, figCols):
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
    
    for r,row in enumerate(multiProjectComparisonD['modelling']['regressionModels']):

        if multiProjectComparisonD['modelling']['regressionModels'][row]['apply']:
            
            regressionModelL.append(row)
            
    figRows = len(regressionModelL)
        
    # Set the columns to include
    if multiProjectComparisonD['modelling']['featureImportance']['apply']:
        
        if multiProjectComparisonD['modelling']['featureImportance']['permutationImportance']['apply']:
        
            multCompPlotsColumnD['permutationImportance'] = len(multCompPlotIndexL)
            multCompPlotIndexL.append('permutationImportance')
        
        if multiProjectComparisonD['modelling']['featureImportance']['treeBasedImportance']['apply']:
        
            multCompPlotsColumnD['treeBasedImportance'] = len(multCompPlotIndexL)
            multCompPlotIndexL.append('treeBasedImportance')
                
        if multiProjectComparisonD['modelling']['featureImportance']['coefficientImportance']['apply']:
        
            multCompPlotsColumnD['coefficientImportance'] = len(multCompPlotIndexL)
            multCompPlotIndexL.append('coefficientImportance')
            
        
            
    if multiProjectComparisonD['modelling']['modelTests']['apply']:
        
        if multiProjectComparisonD['modelling']['modelTests']['trainTest']['apply']:
        
            multCompPlotsColumnD['trainTest'] = len(multCompPlotIndexL)
            multCompPlotIndexL.append('trainTest')
            
        if multiProjectComparisonD['modelling']['modelTests']['Kfold']['apply']:
        
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
            
            print (suptitle)
            #SNULLE
                
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

def PlotOutlierDetect(plotLayout, plotFPN,
         XtrainInliers, XtrainOutliers, XtestInliers, XtestOutliers,  
         postTrainSamples, nTrainOutliers, postTestSamples, nTestOutliers,
         detector, columnsX, targetFeature,  outlierFit, X):
    """
    """

    outliersfig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True, sharey=True )
                
    inScatSymb =  plotLayout.outlierDetection.scatter.inliers
    
    outScatSymb =  plotLayout.outlierDetection.scatter.outliers
    
    if plotLayout.outlierDetection.xlabel == 'auto':
                                
        xlabel = '%s (covariate)' %(columnsX[0])
        
    else:
            
        xlabel = plotLayout.outlierDetection.xlabel
    
    if plotLayout.outlierDetection.ylabel == 'auto':
        
        if columnsX[1] == 'target':
            
            ylabel = '%s (target feature)' %(targetFeature)
        
        else:
            
            ylabel = '%s (covariate)' %(columnsX[1])
            
    else:
        
        ylabel = plotLayout.outlierDetection.xlabel
            
    for i in range (2):
        
        if i == 0:
            inliersX =  XtrainInliers[columnsX[0]]
        
            inliersY =  XtrainInliers[columnsX[1]]
            
            outliersX =  XtrainOutliers[columnsX[0]]
            
            outliersY =  XtrainOutliers[columnsX[1]]
            
            if plotLayout.outlierDetection.annotate.apply:
    
                if plotLayout.outlierDetection.annotate.train == 'auto':
                 
                    txtStr = 'Inlier/outlier samples\n  in: %s, out: %s\n  method: %s' %(postTrainSamples, nTrainOutliers,
                                                detector)
                    
                else:
                    
                    txtStr = plotLayout.outlierDetection.annotate.train
                
            title = 'Outlier detection training dataset'
            
        else:
            
            inliersX =  XtestInliers[columnsX[0]]
        
            inliersY =  XtestInliers[columnsX[1]]
            
            outliersX =  XtestOutliers[columnsX[0]]
            
            outliersY =  XtestOutliers[columnsX[1]]
            
            if plotLayout.outlierDetection.annotate.apply:
    
                if plotLayout.outlierDetection.annotate.test == 'auto':
                 
                    txtStr = 'Inlier/outlier samples\n  in: %s, out: %s\n  method: %s' %(postTestSamples, nTestOutliers,
                                                detector)
                    
                else:
                    
                    txtStr = plotLayout.outlierDetection.annotate.train

            title = 'Outlier detection test dataset'
        
        DecisionBoundaryDisplay.from_estimator(
            outlierFit,
            X,
            response_method="decision_function",
            plot_method="contour",
            colors=plotLayout.outlierDetection.isolines.color,
            levels=[0],
            ax=ax[i],
        )
                         
        ax[i].scatter(inliersX, inliersY, color=inScatSymb.color, alpha=inScatSymb.alpha, s=inScatSymb.size)
        ax[i].scatter(outliersX, outliersY, color=outScatSymb.color, alpha=outScatSymb.alpha, s=outScatSymb.size)
        
        ax[i].set(
            xlabel= xlabel,
            ylabel= ylabel,
            title= title,
        )
        
        ax[i].annotate(txtStr,
               (plotLayout.outlierDetection.annotate.x,
                plotLayout.outlierDetection.annotate.y),
               xycoords = 'axes fraction' )
    
    # Set supTitle
    if plotLayout.outlierDetection.supTitle:
        '''
        if plotLayout.outlierDetection.supTitle == 'auto':
            
            outliersfig.suptitle('Outlier detection and removal')
        
        else:
    
            outliersfig.suptitle(plotLayout.outlierDetection.supTitle)
        '''
           
        if plotLayout.outlierDetection.supTitle:
            
            if '%s' in plotLayout.outlierDetection.supTitle:
                
                suptitle = plotLayout.outlierDetection.supTitle.replace('%s',targetFeature)
                
                outliersfig.suptitle(suptitle)
        
        else:
            
            outliersfig.suptitle(plotLayout.outlierDetection.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        outliersfig.tight_layout() 
                        
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
        
        outliersfig.savefig(plotFPN)
        
def PlotVarianceThreshold(plotLayout, plotFPN,
         X_train, X_test, retainL, discardL, columns, scaler):
    """ 
    """
    
    # TODO: plot variance on right y-axis
    from math import ceil
    
    # Get the plot layout arguments
    maxSpectra = plotLayout.maxSpectra
            
    subplotTitles = plotLayout.varianceThreshold.subplotTitles
    
    # Get the bands to plot
    plotskipStep = ceil( (len(X_train.index)-1)/maxSpectra )
    
    # ttration = trian-test ratio - only for adjusting colorramp
    ttratio = plotskipStep / ceil( (len(X_test.index)-1)/maxSpectra)
        
    trainSelectDF = X_train[ retainL ]
        
    testSelectDF = X_test[ retainL ]
    
    xlabels = list(columns.keys())
            
    xaxislabel = 'wavelength'
        
    yaxislabel = 'reflectance'
    
    if xlabels[0].startswith('pc-'):
        
        xaxislabel = 'principal components'
        
        yaxislabel = 'eigenvalues'
        
    elif xlabels[0].startswith('d'):
                
        yaxislabel = 'derivatives'
    
    # Create empty dict to hold the data
    subplotsD = {}
    
    subplotsD['input'] = {}
    
    subplotsD['selected'] = {}
        
    subplotsD['input']['train'] = {'label': 'Training data (input)',
                                      'DF' : X_train}
    
    subplotsD['input']['test'] = { 'label': 'Test data (input)',
                                      'DF' : X_test}
    
    subplotsD['selected']['train'] = {'label': 'Selected covariates',
                                      'DF' : trainSelectDF}
    
    subplotsD['selected']['test'] = { 'label': 'Selected covariates',
                                      'DF' : testSelectDF}

    #fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(self.spectraPlot.subfigs.figSize.x, self.spectraPlot.subfigs.figSize.y), sharex=True  )
    varianceThresholdPlot, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey='row' )
        
    n = int(len(X_train.index)/plotskipStep)+2
        
    # With n bands known, create the colorRamp
    slicedCM = SetcolorRamp(n, plotLayout.colorRamp)
    
    nCovars = len(retainL)+len(discardL)
    
    annotateStrD = {0:'', 1:'', 2:'',}
   
    if plotLayout.varianceThreshold.annotate.input:          
        if plotLayout.varianceThreshold.annotate.input == 'auto':
            annotateStrD[0] = 'Input bands\n  %s covars\n  showing every %s band' %(nCovars, plotskipStep)
        else:
            annotateStrD[0] = plotLayout.varianceThreshold.annotate.input
            
    if plotLayout.varianceThreshold.annotate.standard:          
        if plotLayout.varianceThreshold.annotate.standard == 'auto':
            annotateStrD[1] = 'Selected bands \n  %s selected; %s discarded\n  showing every %s band' %(len(retainL), len(discardL), plotskipStep)
        else:
            annotateStrD[1] = plotLayout.varianceThreshold.annotate.standard
    
    plotStyle =  GetPlotStyle(plotLayout)
    
    for r, key in enumerate(subplotsD):

        if r == 1:
                
                for c in range(len(subplotsD[key])):
                
                    ax[r][c].set(xlabel=xaxislabel)
        
        for c, subplotkey in enumerate(subplotsD[key]):
            
            plotcols = [item for item in subplotsD[key][subplotkey]['DF'].columns ]
            
            plotcolNr = [columns[key] for key in plotcols]
                     
            ax[r][c].annotate(annotateStrD[r],
                           (plotLayout.varianceThreshold.annotate.x,
                            plotLayout.varianceThreshold.annotate.y),
                           xycoords = 'axes fraction' )
           
            ax[r][c].set( title=subplotsD[key][subplotkey]['label'])
            
            if c == 0:
                
                if scaler == 'None':
                  
                    #ax[r][c].set(ylabel=plotLayout.varianceThreshold.ylabels[r])
                    ax[r][c].set(ylabel=yaxislabel)
                    
                else:
                    
                    ylabel = '%s %s' %(scaler, yaxislabel) 
                    
                    ax[r][c].set(ylabel=ylabel)
                                           
            # Loop over the spectra
            i = -1
        
            n = 0
            
            for index, row in subplotsD[key][subplotkey]['DF'].iterrows():
                
                i += 1
                
                if i % plotskipStep == 0:
                                         
                    if c == 0:

                        ax[r][c].plot(plotcolNr, row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                        
                        if r == 0:
                            
                            ax[1][c].plot(plotcolNr, row, plotStyle, color='grey', ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                    else:
                        
                        m = ceil(n*ttratio)
                        
                        ax[r][c].plot(plotcolNr, row, plotStyle, color=slicedCM[m], ms=plotLayout.pointsize,lw=plotLayout.linewidth) 

                        if r == 0:
                            
                            ax[1][c].plot(plotcolNr, row, plotStyle, color='grey', ms=plotLayout.pointsize,lw=plotLayout.linewidth) 
           
                    n += 1
                      
            if r == 1 and plotLayout.varianceThreshold.axvline:
                
                for xvalue in discardL:
                    
                    ax[1][c].axvline(x=columns[xvalue], color='grey')           
                                   
    # Set supTitle
    if plotLayout.varianceThreshold.supTitle:
        
        if plotLayout.varianceThreshold.supTitle == 'auto':
            
            varianceThresholdPlot.suptitle('Variance threshold covariate selection')
        
        else:
    
            varianceThresholdPlot.suptitle(plotLayout.varianceThreshold.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        varianceThresholdPlot.tight_layout()
 
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
  
        varianceThresholdPlot.savefig(plotFPN)   # save the figure to file
             
    plt.close(varianceThresholdPlot)

def GetAxisLabels(xlabels):
    '''
    '''
    
    xaxislabel = 'wavelength'
        
    yaxislabel = 'reflectance'
    
    if xlabels[0].startswith('pc-'):
        
        xaxislabel = 'principal components'
        
        yaxislabel = 'eigenvalues'
        
    elif xlabels[0].startswith('d'):
                
        yaxislabel = 'derivatives'
    
    return (xaxislabel, yaxislabel)

def SetAxvspan(discardL, columns):
    '''
    '''
    columnKeys = list(columns.keys())

    axvspanD = {}
 
    firstDiscard = True
    
    for i,item in enumerate(columns):
        
        if item in discardL:
            
            if i == 0:
                
                spanBegin = columns[columnKeys[0]]
                #spanEnd = columns[columnKeys[i+1]]
                spanEnd = (columns[columnKeys[i+1]]+columns[columnKeys[i]])/2
                
                
            elif i == len(columns)-1:
                
                spanBegin = (columns[columnKeys[i-1]]+columns[columnKeys[i]])/2
                spanEnd = columns[columnKeys[i]]
                
            else:
                
                #spanBegin = columns[columnKeys[i-1]]
                #spanEnd = columns[columnKeys[i+1]]
                
                spanBegin = (columns[columnKeys[i-1]]+columns[columnKeys[i]])/2
                spanEnd = (columns[columnKeys[i+1]]+columns[columnKeys[i]])/2
                
            if firstDiscard:
            
                axvspanD[item] = {'begin': spanBegin, 'end':spanEnd}
                
                firstDiscard = False
                
                previousDiscard = item
                
            else:
                
                if axvspanD[previousDiscard]['end'] >= spanBegin:
                    
                    axvspanD[previousDiscard]['end'] = spanEnd
                
                else:
               
                    axvspanD[item] = {'begin': spanBegin, 'end':spanEnd}
                    
                    previousDiscard = item
        
    return axvspanD  
                       
def PlotCoviariateSelection(selector, selectorSymbolisation,  plotLayout, plotFPN,
         X_train, X_test, retainL, discardL, columns, targetFeatureName, regressorName='None', scaler='None'):
    """ Plot for all covariate selections
    """
    
    # TODO: plot variance on right y-axis
    from math import ceil
    
    if plotLayout.axvspan.apply:
        
        axvspanD = SetAxvspan(discardL, columns)
        
    # Get the plot layout arguments
    maxSpectra = plotLayout.maxSpectra
            
    subplotTitles = plotLayout.varianceThreshold.subplotTitles
    
    # Get the bands to plot
    plotskipStep = ceil( (len(X_train.index)-1)/maxSpectra )
    
    # ttration = trian-test ratio - only for adjusting colorramp
    ttratio = plotskipStep / ceil( (len(X_test.index)-1)/maxSpectra)
        
    trainSelectDF = X_train[ retainL ]
        
    testSelectDF = X_test[ retainL ]
    
    xlabels = list(columns.keys())
    
    xaxislabel, yaxislabel = GetAxisLabels(xlabels)
    
    # Create empty dict to hold the data
    subplotsD = {}
    
    subplotsD['input'] = {}
    
    subplotsD['selected'] = {}
    
    subplotsD['discarded'] = {}
    
    subplotsD['input']['train'] = {'label': 'Training data (input)',
                                      'DF' : X_train}
    
    subplotsD['input']['test'] = { 'label': 'Test data (input)',
                                      'DF' : X_test}
    
    subplotsD['selected']['train'] = {'label': 'Selected covariates',
                                      'DF' : trainSelectDF}
    
    subplotsD['selected']['test'] = { 'label': 'Selected covariates',
                                      'DF' : testSelectDF}
    
    selectPlotFig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey='row' )
        
    n = int(len(X_train.index)/plotskipStep)+2
        
    # With n bands known, create the colorRamp
    slicedCM = SetcolorRamp(n, plotLayout.colorRamp)
    
    nCovars = len(retainL)+len(discardL)
    
    annotateStrD = {0:'', 1:'', 2:'',}
   
    if selectorSymbolisation.annotate.input:          
        if selectorSymbolisation.annotate.input == 'auto':
 
            annotateStrD[0] = 'Input bands\n  %s covars\n  showing every %s band' %(nCovars, plotskipStep)
          
        else:
            annotateStrD[0] = selectorSymbolisation.annotate.input
    if selectorSymbolisation.annotate.output:
          
              
        if selectorSymbolisation.annotate.output == 'auto':
            if regressorName == 'None':
                annotateStrD[1] = 'Target feature: %s\n   %s selected; %s discarded' %(targetFeatureName, len(retainL), len(discardL))
            else:
                annotateStrD[1] = 'Target feature: %s\n  Regressor: %s\n  %s selected; %s discarded' %(targetFeatureName, regressorName, len(retainL), len(discardL))
        else:
            annotateStrD[1] = selectorSymbolisation.annotate.output
    
    plotStyle =  GetPlotStyle(plotLayout)
    
    for r, key in enumerate(subplotsD):

        if r == 1: # second (last) row - set xaxis label
                
            for c in range(len(subplotsD[key])):
                
                ax[r][c].set(xlabel=xaxislabel)
        
        for c, subplotkey in enumerate(subplotsD[key]):
            
            plotcols = [item for item in subplotsD[key][subplotkey]['DF'].columns ]
            
            plotcolNr = [columns[k] for k in plotcols]
                     
            ax[r][c].annotate(annotateStrD[r],
                           (selectorSymbolisation.annotate.x,
                            selectorSymbolisation.annotate.y),
                           xycoords = 'axes fraction',zorder=4 )
           
            ax[r][c].set( title=subplotsD[key][subplotkey]['label'])
            
            if c == 0:
                
                if scaler == 'None':
                  
                    ax[r][c].set(ylabel=yaxislabel)
                    
                else:
                    
                    ylabel = '%s %s' %(scaler, yaxislabel) 
                    
                    ax[r][c].set(ylabel=ylabel)
                                           
            # Loop over the spectra
            i = -1
        
            n = 0
            
            for index, row in subplotsD[key][subplotkey]['DF'].iterrows():
                
                i += 1
                
                if i % plotskipStep == 0:
                                         
                    if c == 0:

                        ax[r][c].plot(plotcolNr, row, plotStyle, color=slicedCM[n], ms=plotLayout.pointsize,lw=plotLayout.linewidth,zorder=2) 

                        
                        if r == 0:
                            
                            ax[1][c].plot(plotcolNr, row, plotStyle, color='lightgrey', ms=plotLayout.pointsize,lw=plotLayout.linewidth,zorder=1) 

                    else:
                        
                        m = ceil(n*ttratio)
                        
                        ax[r][c].plot(plotcolNr, row, plotStyle, color=slicedCM[m], ms=plotLayout.pointsize,lw=plotLayout.linewidth,zorder=2) 

                        if r == 0:
                            
                            ax[1][c].plot(plotcolNr, row, plotStyle, color='lightgrey', ms=plotLayout.pointsize,lw=plotLayout.linewidth,zorder=1) 
           
                    n += 1
                  
            if r == 1 and plotLayout.axvspan.apply:
                
                for span in axvspanD:
                    
                    ax[1][c].axvspan(axvspanD[span]['begin'], axvspanD[span]['end'], 
                                     ymin=plotLayout.axvspan.ymin, ymax=plotLayout.axvspan.ymax, 
                                     color=plotLayout.axvspan.color, alpha=plotLayout.axvspan.alpha,
                                     zorder=3)           
                                   
    # Set supTitle
    if selectorSymbolisation.supTitle:
        
        if plotLayout.varianceThreshold.supTitle == 'auto':
            
            if regressorName == 'None':
                
                suptitle = '%s covariate selection for %s' %(selector, targetFeatureName, regressorName)
            
            else:
                
                suptitle = '%s covariate selection for %s (regresson: %s)' %(selector, targetFeatureName, regressorName)
            
            selectPlotFig.suptitle(suptitle)
        
        else:
    
            if '%s' in selectorSymbolisation.supTitle:
                
                suptitle = selectorSymbolisation.supTitle.replace('%s',targetFeatureName)
                
                selectPlotFig.suptitle(suptitle)
            
            else:
                
                selectPlotFig.suptitle(selectorSymbolisation.supTitle)
    
    # Set tight layout if requested
    if plotLayout.tightLayout:
    
        selectPlotFig.tight_layout()
                 
    if plotLayout.screenShow:
    
        plt.show()
    
    if plotLayout.savePng:
    
        selectPlotFig.savefig(plotFPN)   # save the figure to file
             
    plt.close(selectPlotFig)   
     
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

        if self.modelPlot.singles.figSize.x == 0:

            self.modelPlot.singles.figSize.x = 8

        if self.modelPlot.singles.figSize.y == 0:

            self.modelPlot.singles.figSize.y = 6

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

        if self.modelPlot.singles.figSize.x == 0:

            self.modelPlot.singles.figSize.x = 8

        if self.modelPlot.singles.figSize.y == 0:

            self.modelPlot.singles.figSize.y = 6

    def _SetModelDefaults(self):
        ''' Set class object default data if required
        '''

        if self.modelPlot.singles.figSize.x == 0:

            self.modelPlot.singles.figSize.x = 4

        if self.modelPlot.singles.figSize.y == 0:

            self.modelPlot.singles.figSize.y = 4

        # Check if Manual feature selection is set
        if self.manualFeatureSelection.apply:

            # Turn off the derivates alteratnive (done as part of the manual selection if requested)
            self.spectraInfoEnhancement.derivatives.apply = False

            # Turn off all other feature selection/agglomeration options
            self.generalFeatureSelection.apply = False

            self.specificFeatureSelection.apply = False

            self.specificFeatureAgglomeration.apply = False

    def __iter__(self):
        '''
        '''
        
        for attr, value in self.__dict__.iteritems():
            yield attr, value

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
        
        
    def _ExtractDataFrameX(self):
        ''' Extract the original dataframe to X (covariate) array and y (predict) column
        '''

        # define the list of covariates to use
        #columnsX = [item for item in self.spectraDF.columns if item not in self.omitL]
        columnsX = self.spectraDF.columns

        columnsY = self.abundanceDf.columns
        
        frames = [self.spectraDF,self.abundanceDf]
    
        spectraDF = pd.concat(frames, axis=1)
        
        self.Xall = spectraDF[columnsX]
        
        columns = self.Xall.columns
        
        if '.' in columns[0]:
            
            XcolumnsNum = [float(item) for item in columns]
            
            XcolumnsStr = ["{0:.1f}".format(item) for item in XcolumnsNum]
            
        else:
            
            XcolumnsNum = [int(item) for item in columns]
            
            XcolumnsStr = list(map(str, XcolumnsNum))
               
        self.Xcolumns = dict(zip(XcolumnsStr,XcolumnsNum))
                
        self.Yall = spectraDF[columnsY]
        
        #Split the data into training and test subsets
        self.X_train, self.X_test, self.Y_train, self.Y_test = model_selection.train_test_split(self.Xall, self.Yall, test_size=self.datasetSplit.testSize)
          
              
    def _SplitDataSetTargetOld(self):
        '''
        '''
        
        #Split the data into training and test subsets
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.Xall, self.y, test_size=self.datasetSplit.testSize)
        
    def _ResetDataFramesXY(self):
        '''
        '''
        
        Xcolumns = list(self.Xall.keys())
        
        Ycolumns = list(self.Yall.keys())
        
        xtrain = np.array(self.X_train)
                    
        xtest = np.array(self.X_test)
                            
        self.X_train = pd.DataFrame(xtrain, columns=Xcolumns)
                    
        self.X_test = pd.DataFrame(xtest, columns=Xcolumns)
        
        ytrain = np.array(self.Y_train)
                    
        ytest = np.array(self.Y_test)
                            
        self.Y_train = pd.DataFrame(ytrain, columns=Ycolumns)
                    
        self.Y_test = pd.DataFrame(ytest, columns=Ycolumns)
        
    def _ExtractDataFrameTarget(self):
        ''' Extract the original dataframe to X (covariate) array and y (predict) column
        '''

        # Extract the target feature
        self.y = self.abundanceDf[self.targetFeature]

        # Append the target array to the self.spectraDF dataframe
        self.spectraDF['target'] = self.y

        # define the list of covariates to use
        #columnsX = [item for item in self.spectraDF.columns if item not in self.omitL]
        columnsX = [item for item in self.spectraDF.columns]

        # extract all the covariate columns as Xll
        self.Xall = self.spectraDF[columnsX]
        
        # Drop the added target column from the dataframe
        self.spectraDF = self.spectraDF.drop('target', axis=1)

        # Remove all samples where the targetfeature is NaN
        self.Xall = self.Xall[~np.isnan(self.Xall).any(axis=1)]
        
        # Drop the added target column from self.X
        self.Xall = self.Xall.drop('target', axis=1)

        # Then also delete NaN from self.y
        self.y = self.y[~np.isnan(self.y)]
        
        # Remove all non-finite values   
        self.Xall = self.Xall[np.isfinite(self.y)] 
        
        self.y = self.y[np.isfinite(self.y)] 
     
    
    def _Filtering(self, extractionMode, beginWL, endWL, outputBandWidth, wlArr, halfwlstep):
        """ Filtering the spectral signal
        """
                
        filtertxt = 'None'
        
        # Get the input spectra
        columnsX = [item for item in self.spectraDF.columns]
                
        outputWls = [float(item) for item in columnsX]
       
        # extract the covariate columns as X
        X = self.spectraDF[columnsX]
        
        # If the filter is a kernel (including moving average)    
        if self.spectraPreProcess.filtering.movingaverage.kernel:
 
            sumkernel =  np.asarray(self.spectraPreProcess.filtering.movingaverage.kernel).sum()
            
            normkernel = self.spectraPreProcess.filtering.movingaverage.kernel/sumkernel
            
            X1 = convolve1d(X, normkernel, axis=-1, mode=self.spectraPreProcess.filtering.movingaverage.mode)
            
            filtertxt = 'kernel filter'
                 
        # If Gaussian filter
        elif self.spectraPreProcess.filtering.Gauss.sigma:
            
            sigma = self.spectraPreProcess.filtering.Gauss.sigma / ( (wlArr[len(wlArr)-1] - wlArr[0])/(len(wlArr)-1) )
                                    
            X1 = gaussian_filter1d(X, sigma, axis=-1, mode=self.spectraPreProcess.filtering.Gauss.mode)
                           
            filtertxt = 'Gaussian filter'
             
        # If Savitzky Golay filter
        elif self.spectraPreProcess.filtering.SavitzkyGolay.window_length:
            
            X1 = savgol_filter(X, window_length=self.spectraPreProcess.filtering.SavitzkyGolay.window_length, 
                               polyorder= self.spectraPreProcess.filtering.SavitzkyGolay.polyorder,
                               axis=-1,
                               mode=self.spectraPreProcess.filtering.SavitzkyGolay.mode)
                        
            filtertxt = 'Savitzky-Golay filter'
            
        else:
            
            X1 = X
                        
        if extractionMode.lower() in ['none', 'no'] or beginWL >= endWL:
            
            "No extraction"
            
            pass
            
        else:
            
            # Define the output wavelengths
            if extractionMode == 'noendpoints':
                
                outputWls = np.arange(beginWL+outputBandWidth, endWL, outputBandWidth)
                
            else:
                    
                outputWls = np.arange(beginWL, endWL, outputBandWidth)
                            
            xDF = pd.DataFrame(data=X1, columns=columnsX)
            
            arrL = []
            
            for row in xDF.values:
  
                spectraA = np.interp(outputWls, wlArr, row,
                            left=beginWL-halfwlstep,
                            right=endWL+halfwlstep)
                
                arrL.append(spectraA)
                
            X1 = np.asarray(arrL)
                                       
        return filtertxt, outputWls, X1
    
    def _FilterPrep(self):
        '''
        '''
        # Create an empty copy of the spectra DataFrame
        originalDF = self.spectraDF.copy()
                
        columnsNum = list(self.columns.values())
          
        # Convert bands to array)
        wlArr = np.asarray(columnsNum)
        
        halfwlstep = self.spectraPreProcess.filtering.extraction.outputBandWidth/2
        
        extractionMode = self.spectraPreProcess.filtering.extraction.mode
        
        beginWL = self.spectraPreProcess.filtering.extraction.beginWaveLength
        
        endWL = self.spectraPreProcess.filtering.extraction.endWaveLength-1
        
        outputBandWidth = self.spectraPreProcess.filtering.extraction.outputBandWidth
        
        # Run the filtering
        filtertxt, outputWls, X1 = self._Filtering(extractionMode, beginWL, endWL, outputBandWidth, wlArr, halfwlstep)
        
        if isinstance(outputWls[0], Integral):
            
            #outputWls = ["{d}".format(item) for item in outputWls]
            outputWlsStr = list(map(str, outputWls))

        else:
            
            outputWlsStr = ["{0:.1f}".format(item) for item in outputWls]
            
        # Reset self.columns
        self.columns = dict(zip(outputWlsStr,outputWls))
        
        self.spectraDF = pd.DataFrame(data=X1, columns=outputWlsStr)
        
        if self.enhancementPlotLayout.filterExtraction.apply:
            
            PlotFilterExtract( self.enhancementPlotLayout, filtertxt, originalDF, self.spectraDF, self.filterExtractPlotFPN)
            
        self.spectraDF = pd.DataFrame(data=X1, columns=outputWlsStr)
                    
        return filtertxt
                
    def _MultiFiltering(self):
        ''' Applies different filters over different parts of the spectra
        '''
        
        # Create an empty copy of the spectra DataFrame
        newSpectraDF = self.spectraDF[[]].copy()
             
        # Extract the columns bands) as floating wavelenhts   
        columnsX = [float(item) for item in self.spectraDF.columns]
        
        # Cnovert bands to array)
        wlArr = np.asarray(columnsX)
        
        outPutWlStr = []
        
        outPutWlNum = []
        
        # Loop over the wavelength regions defined for filtering
        for r, rang in enumerate(self.spectraPreProcess.multifiltering.beginWaveLength):
            
            # Deep copy the spectra DataFrame
            copySpectraDF = deepcopy(self.spectraDF)

            # Set all the filteroptions to False before starting each loop
            self.spectraPreProcess.filtering.movingaverage.kernel = []
            self.spectraPreProcess.filtering.Gauss.sigma = 0
            self.spectraPreProcess.filtering.SavitzkyGolay.window_length = 0
                        
            if self.spectraPreProcess.multifiltering.movingaverage.kernel[r]:
                
                self.spectraPreProcess.filtering.movingaverage.kernel = self.spectraPreProcess.multifiltering.movingaverage.kernel[r]
                
            elif self.spectraPreProcess.multifiltering.SavitzkyGolay.window_length[r]:
                                
                self.spectraPreProcess.filtering.SavitzkyGolay.window_length = self.spectraPreProcess.multifiltering.SavitzkyGolay.window_length[r]
                
            elif self.spectraPreProcess.multifiltering.Gauss.sigma[r]:
                                
                self.spectraPreProcess.filtering.Gauss.sigma = self.spectraPreProcess.multifiltering.Gauss.sigma[r]
             
            halfwlstep = self.spectraPreProcess.multifiltering.outputBandWidth[r]/2
            extractionMode = self.spectraPreProcess.filtering.extraction.mode
            beginWL = self.spectraPreProcess.multifiltering.beginWaveLength[r]
            endWL = self.spectraPreProcess.multifiltering.endWaveLength[r]-1
            outputBandWidth = self.spectraPreProcess.multifiltering.outputBandWidth[r]
               
            # Run the filtering
            filtertxt, outputWls, X1 = self._Filtering(extractionMode, beginWL, endWL, outputBandWidth, wlArr, halfwlstep)
            
            if isinstance(outputWls[0], Integral):
            
                #outputWls = ["{d}".format(item) for item in outputWls]
                outputWlS = list(map(str, outputWls))
    
            else:
                    
                outputWlS = ["{0:.1f}".format(item) for item in outputWls]
                    
            outPutWlNum.extend(outputWls)
            
            outPutWlStr.extend(outputWlS)
        
            # Add filtered+extrad range with columns as strings
            newSpectraDF[ outputWlS ] = X1
           
        # reset columns 
        self.columns = dict(zip(outPutWlStr, outPutWlNum))
                            
        self.spectraDF = deepcopy(copySpectraDF)
            
        if self.enhancementPlotLayout.filterExtraction.apply:
            
            PlotFilterExtract( self.enhancementPlotLayout, filtertxt, self.spectraDF, newSpectraDF, self.filterExtractPlotFPN)

        # Set the fitlered spectra to spectraDF
        self.spectraDF = newSpectraDF

        return 'multi'   

        
    def _ResetRegressorXyDF(self):
        '''
        '''

        self.X_train_R =  deepcopy(self.X_train_T)
        self.X_test_R =  deepcopy(self.X_test_T)
        self.y_train_r =  deepcopy(self.y_train_t)
        self.y_test_r =  deepcopy(self.y_test_t)
        
        self.X_columns_R = deepcopy(self.X_columns_T)
        
        # Reomve all NoN and infinity
        self._ResetXY_R()
                       
    
            
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

        #Fit the model
        model.fit(self.X_train_R, self.y_train_r)

        #Predict the independent variable in the test subset
        predict = model.predict(self.X_test_R)
        
        r2_total = r2_score(self.y_test_r, predict)
        
        rmse_total = sqrt(mean_squared_error(self.y_test_r, predict))
        
        medae_total = median_absolute_error(self.y_test_r, predict)
        
        mae_total = mean_absolute_error(self.y_test_r, predict)
        
        mape_total = mean_absolute_percentage_error(self.y_test_r, predict)
        
        

        self.trainTestResultD[self.targetFeature][name] = {'rmse':rmse_total,
                                                           'mae':mae_total,
                                                           'medae': medae_total,
                                                           'mape':mape_total,
                                                           'r2': r2_total,
                                                           'hyperParameterSetting': self.paramD['modelling']['regressionModels'][name]['hyperParams'],
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

            infoStr =  '\n                trainTest Model: %s\n' %(name)
            
            infoStr += '                    Root mean squared error (RMSE) total: %.2f\n' % rmse_total
            infoStr += '                    Variance (r2) score total: %.2f\n' % r2_total
            
            if self.verbose > 1:

                infoStr += '                    Mean absolute error (MAE) total: %.2f\n' %( mae_total)
    
                infoStr += '                    Mean absolute percent error (MAPE) total: %.2f\n' %( mape_total)
    
                infoStr += '                    Median absolute error (MedAE) total: %.2f\n' %( medae_total)
                            
                infoStr += '                    hyperParams: %s\n' %(self.paramD['modelling']['regressionModels'][name]['hyperParams'])
            
            print (infoStr)
                    
        if self.modelPlot.apply:
            txtstr = 'nspectra: %s\n' %(self.Xall.shape[0])
            txtstr += 'nbands: %s\n' %(self.Xall.shape[1])
            #txtstr += 'min wl: %s\n' %(self.bandL[0])
            #txtstr += 'max wl: %s\n' %(self.bandL[len(self.bandL)-1])
            #txtstr += 'bands: %s\n' %( ' ,'.join('({0})'.format(w) for w in self.aggBandL)  )
            #txtstr += 'width wl: %s' %(int(self.bandL[1]-self.bandL[0]))

            #txtstrHyperParams =  self.HPtuningtxt+'\nHyper Parameters:\n'+'\n'.join([key+': '+str(value) for key, value in self.tunedModD[name].items()])
            suptitle = '%s train/test model (testsize = %s)' %(self.targetFeature, self.datasetSplit.testSize)
            title = ('Model: %(mod)s; RMSE: %(rmse)2f; r2: %(r2)2f' \
                      % {'mod':name,'rmse':mean_squared_error(self.y_test_r, predict),'r2': r2_score(self.y_test_r, predict)} )

            txtstr = ('RMSE: %(rmse)2f\nr2: %(r2)2f\nnTrain: %(i)d\nnTest: %(j)d' \
                      % {'rmse':self.trainTestResultD[self.targetFeature][name]['rmse'],
                         'r2': self.trainTestResultD[self.targetFeature][name]['r2'],
                         'i': self.X_train_R.shape[0], 'j': self.X_test_R.shape[0]})

            self._PlotRegr(self.y_test_r, predict, suptitle, title, txtstr, '',name, 'trainTest', multCompAxs)

    def _RegrModKFold(self, multCompAxs):
        """
        """

        #Retrieve the model name and the model itself
        name,model = self.regrModel

        predict = model_selection.cross_val_predict(model, self.X, self.y, cv=self.modelling.modelTests.Kfold.folds)

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
                                                        'hyperParameterSetting': self.paramD['modelling']['regressionModels'][name]['hyperParams'],
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

            infoStr =  '\n                Kfold Model: %s\n' %(name)
            
            infoStr += '                    Root mean squared error (RMSE) total: %.2f\n' % rmse_total
            
            infoStr += '                    Variance (r2) score total: %.2f\n' % r2_total
            
            if self.verbose > 1:
            
                infoStr += '                    RMSE folded: %.2f (%.2f) \n' %( -1*rmse_folded.mean(),  rmse_folded.std())
                
                infoStr += '                    Mean absolute error (MAE) folded: %.2f (%.2f) \n' %( -1*mae_folded.mean(),  mae_folded.std())
    
                infoStr += '                    Mean absolute percent error (MAPE) folded: %.2f (%.2f) \n' %( -1*mape_folded.mean(),  mape_folded.std())
    
                infoStr += '                    Median absolute error (MedAE) folded: %.2f (%.2f) \n' %( -1*medae_folded.mean(),  medae_folded.std())
    
                infoStr += '                    Variance (r2) score folded: %.2f (%.2f) \n' %( r2_folded.mean(),  r2_folded.std())

                infoStr += '                    hyperParams: %s\n' %(self.paramD['modelling']['regressionModels'][name]['hyperParams'])

            print (infoStr)
            
        txtstr = 'nspectra: %s\n' %(self.X.shape[0])
        txtstr += 'nbands: %s\n' %(self.X.shape[1])
        #txtstr += 'min wl: %s\n' %(self.bandL[0])
        #txtstr += 'max wl: %s\n' %(self.bandL[len(self.bandL)-1])
        #txtstr += 'bands: %s\n' %( ' ,'.join('({0})'.format(w) for w in self.aggBandL)  )
        #txtstr += 'width wl: %s' %(int(self.bandL[1]-self.bandL[0]))

        #txtstrHyperParams =  self.HPtuningtxt+'\nHyper Parameters:\n'+'\n'.join([key+': '+str(value) for key, value in self.tunedModD[name].items()])
        suptitle = '%s Kfold model (nfolds = %s)' %(self.targetFeature, self.modelling.modelTests.Kfold.folds)
        title = ('Model: %(mod)s; RMSE: %(rmse)2f; r2: %(r2)2f' \
                  % {'mod':name,'rmse':rmse_total,'r2': r2_total} )

        txtstr = ('RMSE: %(rmse)2f\nr2: %(r2)2f\nSamples: %(n)d' \
                      % {'rmse':self.KfoldResultD[self.targetFeature][name]['rmse_total'],
                         'r2': self.KfoldResultD[self.targetFeature][name]['r2_total'],
                         'n': self.X.shape[0]} )

        self._PlotRegr(self.y, predict, suptitle, title, txtstr, '',name, 'Kfold', multCompAxs)

    def _PyplotArgs(self, pyplotAttrs, argD):
                        
        args = [a for a in dir(self.featureImportancePlot.xticks) if not a.startswith('_')]
        
        argD = {"axis":"x"}
        
        for arg in args:
            
            val = '%s' %(getattr(pyplotAttrs, arg))
            
            if (val.replace('.','')).isnumeric():
                
                argD[arg] = getattr(pyplotAttrs, arg)
            
            else:

                argD[arg] = val
                
        return argD
    
    def _SetTargetFeatureSymbol(self):
        '''
        '''

        self.featureSymbolColor = 'black'
        
        self.featureSymbolAlpha = 0.1

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
        
        if isinstance(obs,pd.DataFrame):
            
            exit('obs is a dataframe, must be a dataseries')

        if self.modelPlot.singles.apply:
            
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
            ax.text(obs.min(), (obs.max()-obs.min())*0.8, txtstr, fontdict=None,  wrap=True)

            if self.modelPlot.singles.screenShow:

                plt.show()

            if self.modelPlot.singles.savePng:
                
                self.figLibL.append(self.imageFPND[self.targetFeature][regrModel][modeltest])

                fig.savefig(self.imageFPND[self.targetFeature][regrModel][modeltest])
                
            plt.close(fig=fig)


        if self.modelPlot.rows.apply:

            if self.modelPlot.rows.targetFeatures.apply:

                # modeltest is either trainTest of Kfold
                if modeltest in self.modelPlot.rows.targetFeatures.columns:

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

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest] ].scatter(obs, pred, edgecolors=(0, 0, 0),  
                               color=self.featureSymbolColor, alpha=self.featureSymbolAlpha,
                               s=self.paramD['regressionModelSymbols'][self.regrModel[0]]['size'],
                               marker=self.paramD['regressionModelSymbols'][self.regrModel[0]]['marker'])

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest] ].plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest]].text(.05, .95,
                                                        txtstr, ha='left', va='top',
                                                        transform=self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest]].transAxes)

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[modeltest]].yaxis.set_label_position("right")

                    # if at last column
                    if self.targetFeaturePlotColumnD[modeltest] == len(self.modelPlot.rows.regressionModels.columns)-1:

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

            if self.modelPlot.rows.regressionModels.apply:

                # modeltest is either trainTest of Kfold
                if modeltest in self.modelPlot.rows.regressionModels.columns:

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
                    if self.regressionModelPlotColumnD[modeltest] == len(self.modelPlot.rows.targetFeatures.columns)-1:

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
                    '''
                    else:

                        if (len(self.regressorModels)) == 1:

                            self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[modeltest]].set_xlabel('Observations')

                        else:

                            self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[modeltest]].set_xlabel('Observations')
                    '''
                  
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

        if hasattr(self.modelling.regressionModels, 'OLS') and self.modelling.regressionModels.OLS.apply:

            self.regressorModels.append(('OLS', linear_model.LinearRegression(**self.paramD['modelling']['regressionModels']['OLS']['hyperParams'])))

            self.modelSelectD['OLS'] = []

        if hasattr(self.modelling.regressionModels, 'TheilSen') and self.modelling.regressionModels.TheilSen.apply:

            self.regressorModels.append(('TheilSen', linear_model.TheilSenRegressor(**self.paramD['modelling']['regressionModels']['OLS']['hyperParams'])))

            self.modelSelectD['TheilSen'] = []

        if hasattr(self.modelling.regressionModels, 'Huber') and self.modelling.regressionModels.Huber.apply:

            self.regressorModels.append(('Huber', linear_model.HuberRegressor(**self.paramD['modelling']['regressionModels']['OLS']['hyperParams'])))

            self.modelSelectD['Huber'] = []

        if hasattr(self.modelling.regressionModels, 'KnnRegr') and self.modelling.regressionModels.KnnRegr.apply:
            self.regressorModels.append(('KnnRegr', KNeighborsRegressor( **self.paramD['modelling']['regressionModels']['KnnRegr']['hyperParams'])))
            self.modelSelectD['KnnRegr'] = []

        if hasattr(self.modelling.regressionModels, 'DecTreeRegr') and self.modelling.regressionModels.DecTreeRegr.apply:
            self.regressorModels.append(('DecTreeRegr', DecisionTreeRegressor(**self.paramD['modelling']['regressionModels']['DecTreeRegr']['hyperParams'])))
            self.modelSelectD['DecTreeRegr'] = []

        if hasattr(self.modelling.regressionModels, 'SVR') and self.modelling.regressionModels.SVR.apply:
            self.regressorModels.append(('SVR', SVR(**self.paramD['modelling']['regressionModels']['SVR']['hyperParams'])))
            self.modelSelectD['SVR'] = []

        if hasattr(self.modelling.regressionModels, 'RandForRegr') and self.modelling.regressionModels.RandForRegr.apply:
            self.regressorModels.append(('RandForRegr', RandomForestRegressor( **self.paramD['modelling']['regressionModels']['RandForRegr']['hyperParams'])))
            self.modelSelectD['RandForRegr'] = []

        if hasattr(self.modelling.regressionModels, 'MLP') and self.modelling.regressionModels.MLP.apply:

            '''
            # First make a pipeline with standardscaler + MLP
            mlp = make_pipeline(
                StandardScaler(),
                MLPRegressor( **self.paramD['modelling']['regressionModels']['MLP']['hyperParams'])
            )
            '''
            mlp = Pipeline([('scl', StandardScaler()),
                    ('clf', MLPRegressor( **self.paramD['modelling']['regressionModels']['MLP']['hyperParams']) ) ])

            # Then add the pipeline as MLP
            self.regressorModels.append(('MLP', mlp))

            self.modelSelectD['MLP'] = []
        
        if hasattr(self.modelling.regressionModels, 'Cubist') and self.modelling.regressionModels.Cubist.apply:
            self.regressorModels.append(('Cubist', Cubist( **self.paramD['modelling']['regressionModels']['Cubist']['hyperParams'])))
            self.modelSelectD['Cubist'] = []
        '''    
        if hasattr(self.modelling.regressionModels, 'PLS') and self.modelling.regressionModels.PLS.apply:
            self.regressorModels.append(('PLS', PLSRegressor( **self.paramD['modelling']['regressionModels']['PLS']['hyperParams'])))
            self.modelSelectD['RandForRegr'] = []
        '''
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
            
        if hasattr(self, 'featureImportancePlot') and hasattr(self.featureImportancePlot, 'xticks'):
            
            argD = self._PyplotArgs(self.featureImportancePlot.xticks,{"axis":"x"} )
                                
            ax.tick_params(**argD)
                    
        ax.set_title(title)
        
        ax.set_ylim(ymin=0)

        if xyLabel[0]:

            ax.set_ylabel(xyLabel[0])

        if xyLabel[1]:

            ax.set_ylabel(xyLabel[1])

        if self.modelPlot.tightLayout:

            singlefig.tight_layout()

        if self.modelPlot.singles.screenShow:

            plt.show()

        if self.modelPlot.singles.savePng:

            singlefig.savefig(pngFPN)

        plt.close(fig=singlefig)

    def _PlotFeatureImportanceRows(self, featureArray, importanceArray, errorArray, importanceCategory, yLabel):
        '''
        '''

        nnFS = self.X_train.shape

        text = 'tot covars: %s' %(nnFS[1])

        if self.specificFeatureSelectionTxt != None:

            text += '\n%s' %(self.specificFeatureSelectionTxt)

        if self.generalFeatureSelectTxt != None:

            text += '\n%s' %(self.generalFeatureSelectTxt)

        if self.modelPlot.rows.targetFeatures.apply:

            if importanceCategory in self.modelPlot.rows.targetFeatures.columns:

                if (len(self.targetFeatures)) == 1:

                    self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory] ].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

                    self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory] ].tick_params(labelleft=False)

                    self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].text(.3, .95, text, ha='left', va='top',
                                                    transform=self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].transAxes)

                    self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].set_ylabel(yLabel)

                    if hasattr(self, 'featureImportancePlot') and hasattr(self.featureImportancePlot, 'xticks'):
            
                        argD = self._PyplotArgs(self.featureImportancePlot.xticks,{"axis":"x"} )
                                
                        #self.columnAxs[self.regrModel[0]].tick_params(**argD)
                        
                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].tick_params(**argD)

                    self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].set_ylim(ymin=0) 
                    #self.columnAxs[self.regrModel[0]].set_ylim(ymin=0)

                else:
                    

                    self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory] ].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

                    self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory] ].tick_params(labelleft=False)

                    self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].text(.3, .95, text, ha='left', va='top',
                                                    transform=self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].transAxes)

                    self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].set_ylabel(yLabel)
                                           
                    if hasattr(self, 'featureImportancePlot') and hasattr(self.featureImportancePlot, 'xticks'):
            
                        argD = self._PyplotArgs(self.featureImportancePlot.xticks,{"axis":"x"} )
                                
                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].tick_params(**argD)

                    self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].set_ylim(ymin=0)
                    
                if importanceCategory == 'featureImportance':

                    if (len(self.targetFeatures)) == 1:

                        # Draw horisontal line ay y=y
                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].axhline(y=0, lw=1, c='black')

                    else:

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].axhline(y=0, lw=1, c='black')
              
                # if at last row
                if self.targetN == self.nTargetFeatures-1:

                    if (len(self.targetFeatures)) == 1:

                        self.columnAxs[self.regrModel[0]][self.targetFeaturePlotColumnD[importanceCategory]].set_xlabel('Features')

                    else:

                        self.columnAxs[self.regrModel[0]][self.targetN, self.targetFeaturePlotColumnD[importanceCategory]].set_xlabel('Features')

        if self.modelPlot.rows.regressionModels.apply:

            if importanceCategory in self.modelPlot.rows.regressionModels.columns:

                if (len(self.regressorModels)) == 1:

                    self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory] ].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

                    self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory] ].tick_params(labelleft=False)

                    self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].text(.3, .95, text, ha='left', va='top',
                                                transform=self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].transAxes)

                    self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].set_ylabel(yLabel)

                    if hasattr(self, 'featureImportancePlot') and hasattr(self.featureImportancePlot, 'xticks'):
            
                        argD = self._PyplotArgs(self.featureImportancePlot.xticks,{"axis":"x"} )
                                
                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].tick_params(**argD)

                    self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].set_ylim(ymin=0)
                    
                else:

                    self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory] ].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

                    self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory] ].tick_params(labelleft=False)

                    self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].text(.3, .95, text, ha='left', va='top',
                                                    transform=self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].transAxes)

                    self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].set_ylabel(yLabel)

                    if hasattr(self, 'featureImportancePlot') and hasattr(self.featureImportancePlot, 'xticks'):
            
                        argD = self._PyplotArgs(self.featureImportancePlot.xticks,{"axis":"x"} )
                                
                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].tick_params(**argD)
                    
                    self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].set_ylim(ymin=0)
                            
                if importanceCategory == 'featureImportance':

                    if (len(self.regressorModels)) == 1:

                        # Draw horisontal line ay y=y
                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].axhline(y=0, lw=1, c='black')

                    else:

                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].axhline(y=0, lw=1, c='black')
   
                # if at last row set the x-axis label
                if self.regrN == self.nRegrModels-1:

                    if (len(self.regressorModels)) == 1:

                        self.columnAxs[self.targetFeature][self.regressionModelPlotColumnD[importanceCategory]].set_xlabel('Features')

                    else:

                        self.columnAxs[self.targetFeature][self.regrN, self.regressionModelPlotColumnD[importanceCategory]].set_xlabel('Features')

    def _MultCompPlotFeatureImportance(self, featureArray, importanceArray, errorArray, index, yLabel, multCompAxs):
        '''
        '''

        nnFS = self.Xall.shape

        text = 'tot covars: %s' %(nnFS[1])

        if self.specificFeatureSelectionTxt != None:

            text += '\n%s' %(self.specificFeatureSelectionTxt)

        if self.agglomerateTxt != None:

            text += '\n%s' %(self.agglomerateTxt)

        if self.specificFeatureSelectionTxt != None:

            text += '\n%s' %(self.specificFeatureSelectionTxt)
            
        columnNr = self.modelNr
           
        if (len(self.regressorModels)) == 1: # only state the column
            
            multCompAxs[self.targetFeature][index][columnNr].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

            multCompAxs[self.targetFeature][index][columnNr].tick_params(labelleft=False)

            multCompAxs[self.targetFeature][index][columnNr].text(.3, .95, text, ha='left', va='top',
                                        transform=multCompAxs[self.targetFeature][index][columnNr].transAxes)

            #multCompAxs[self.targetFeature][index][columnNr].set_ylabel(yLabel)
            
            if hasattr(self, 'featureImportancePlot') and hasattr(self.featureImportancePlot, 'xticks'):
        
                argD = self._PyplotArgs(self.featureImportancePlot.xticks,{"axis":"x"} )
                            
                multCompAxs[self.targetFeature][index][columnNr].tick_params(**argD)
                
            multCompAxs[self.targetFeature][index][columnNr].set_ylim(ymin=0)


        else:
                            
            multCompAxs[self.targetFeature][index][self.regrN, columnNr].bar(featureArray, importanceArray, yerr=errorArray, color=self.featureSymbolColor)

            multCompAxs[self.targetFeature][index][self.regrN, columnNr].tick_params(labelleft=False)

            multCompAxs[self.targetFeature][index][self.regrN, columnNr].text(.3, .95, text, ha='left', va='top',
                                            transform=multCompAxs[self.targetFeature][index][self.regrN, columnNr].transAxes)

            #multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_ylabel(yLabel)

            if hasattr(self, 'featureImportancePlot') and hasattr(self.featureImportancePlot, 'xticks'):
        
                argD = self._PyplotArgs(self.featureImportancePlot.xticks,{"axis":"x"} )
                            
                multCompAxs[self.targetFeature][index][self.regrN, columnNr].tick_params(**argD)
            
            multCompAxs[self.targetFeature][index][self.regrN, columnNr].set_ylim(ymin=0)
            
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
        
        def FeatureImp():
            '''
            '''
            if name in ['OLS','TheilSen','Huber', "Ridge", "ElasticNet", 'logistic', 'SVR']:

                if name in ['logistic','SVR']:
    
                    importances = model.coef_[0]
    
                else:
    
                    importances = model.coef_
    
                absImportances = abs(importances)
    
                sorted_idx = absImportances.argsort()
    
                importanceArray = importances[sorted_idx][::-1][0:maxFeatures]
    
                featureArray = np.asarray(columns)[sorted_idx][::-1][0:maxFeatures]
    
                featImpD = {}
    
                for i in range(len(featureArray)):
    
                    featImpD[featureArray[i]] = {'linearCoefficient': round(importanceArray[i],4)}
                  
                self.modelFeatureImportanceD[self.targetFeature][name]['featureImportance'] = featImpD
    
                if self.modelPlot.singles.apply:
    
                    title = "Linear feature coefficients\n Feature: %s; Model: %s" %(self.targetFeature, name)
    
                    xyLabels = ['Features','Coefficient']
    
                    pngFPN = self.imageFPND[self.targetFeature][name]['featureImportance']['regressionImportance']
                                        
                    self._PlotFeatureImportanceSingles(featureArray, np.absolute(importanceArray), None, title, xyLabels, pngFPN)
    
                if self.modelPlot.rows.apply:
    
                    self._PlotFeatureImportanceRows(featureArray, np.absolute(importanceArray), None, 'coefficientImportance','rel. coef. weight')
    
                if self.multcompplot:
          
                    self._MultCompPlotFeatureImportance(featureArray, np.absolute(importanceArray), None, 'coefficientImportance', 'rel. coef. weight', multCompAxs)
    
            elif name in ['KnnRegr','MLP', 'Cubist']:
                ''' These models do not have any feature importance to report
                '''
                pass
    
            else:
    
                featImpD = {}
    
                importances = model.feature_importances_
    
                sorted_idx = importances.argsort()
    
                importanceArray = importances[sorted_idx][::-1][0:maxFeatures]
    
                featureArray = np.asarray(columns)[sorted_idx][::-1][0:maxFeatures]
    
                if name in ['RandForRegr']:
    
                    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    
                    errorArray = std[sorted_idx][::-1][0:maxFeatures]
    
                    for i in range(len(featureArray)):
    
                        featImpD[featureArray[i]] = {'MDI': round(importanceArray[i],4),
                                                     'std': round(errorArray[i],4)}
                       
                else:
    
                    errorArray = None
    
                    for i in range(len(featureArray)):
    
                        featImpD[featureArray[i]] = {'MDI': importanceArray[i]}
                        
                        featImpD[featureArray[i]] = {k:(round(v,4) if isinstance(v,float) else v) for (k,v) in featImpD[featureArray[i]]}
    
                self.modelFeatureImportanceD[self.targetFeature][name]['featureImportance'] = featImpD
    
                if self.modelPlot.singles.apply:
    
                    title = "MDI feature importance\n Feature: %s; Model: %s" %(self.targetFeature, name)
    
                    xyLabel = ['Features', 'Mean impurity decrease']
    
                    pngFPN = self.imageFPND[self.targetFeature][name]['featureImportance']['regressionImportance']
    
                    self._PlotFeatureImportanceSingles(featureArray, importanceArray, errorArray, title, xyLabel, pngFPN)
    
                if self.modelPlot.rows.apply:
    
                    self._PlotFeatureImportanceRows(featureArray, importanceArray, errorArray, 'featureImportance', 'rel. mean impur. decr.')

        
        def PermImp():
            '''
            '''
            n_repeats = self.modelling.featureImportance.permutationRepeats

            permImportance = permutation_importance(model, self.X_test_R, self.y_test_r, n_repeats=n_repeats)
    
            permImportanceMean = permImportance.importances_mean
    
            permImportanceStd = permImportance.importances_std
    
            sorted_idx = permImportanceMean.argsort()
    
            permImportanceArray = permImportanceMean[sorted_idx][::-1][0:maxFeatures]
            
            errorArray = permImportanceStd[sorted_idx][::-1][0:maxFeatures]
    
            featureArray = np.asarray(columns)[sorted_idx][::-1][0:maxFeatures]
    
            permImpD = {}
    
            for i in range(len(featureArray)):
    
                permImpD[featureArray[i]] = {'mean_accuracy_decrease': round(permImportanceArray[i],4),
                                             'std': round(errorArray[i],4)}
                
            self.modelFeatureImportanceD[self.targetFeature][name]['permutationsImportance'] = permImpD
    
            if self.modelPlot.singles.apply:
    
                title = "Permutation importance\n Feature: %s; Model: %s" %(self.targetFeature, name)
    
                xyLabel = ['Features', 'Mean accuracy decrease']
    
                pngFPN = self.imageFPND[self.targetFeature][name]['featureImportance']['permutationImportance']
    
                self._PlotFeatureImportanceSingles(featureArray, permImportanceArray, errorArray, title, xyLabel, pngFPN)
                
            if self.modelPlot.rows.apply:
    
                self._PlotFeatureImportanceRows(featureArray, permImportanceArray, errorArray, 'permutationImportance', 'rel. Mean accur. decr.')
    
            if self.multcompplot:
          
                self._MultCompPlotFeatureImportance(featureArray, permImportanceArray, errorArray, 'permutationImportance', 'rel. Mean accur. decr.', multCompAxs)

        
        def TreeBasedImp():
            '''
            '''
 
            forest = RandomForestRegressor(random_state=0)

            forest.fit(self.X_test_R, self.y_test_r)
        
            treeBasedImportanceMean = forest.feature_importances_
     
            treeBasedImportanceStd = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    
            sorted_idx = treeBasedImportanceMean.argsort()
    
            treeBasedImportanceArray = treeBasedImportanceMean[sorted_idx][::-1][0:maxFeatures]
            
            errorArray = treeBasedImportanceStd[sorted_idx][::-1][0:maxFeatures]
    
            featureArray = np.asarray(columns)[sorted_idx][::-1][0:maxFeatures]
    
            treeBasedImpD = {}
    
            for i in range(len(featureArray)):
    
                treeBasedImpD[featureArray[i]] = {'mean_accuracy_decrease': round(treeBasedImportanceArray[i],4),
                                             'std': round(errorArray[i],4)}
                
            self.modelFeatureImportanceD[self.targetFeature][name]['treeBasedImportance'] = treeBasedImpD
    
            if self.modelPlot.singles.apply:
    
                title = "Tree Based importance\n Feature: %s; Model: %s" %(self.targetFeature, name)
    
                xyLabel = ['Features', 'Mean impure. decr.']
    
                pngFPN = self.imageFPND[self.targetFeature][name]['featureImportance']['treeBasedImportance']
    
                self._PlotFeatureImportanceSingles(featureArray, treeBasedImportanceArray, errorArray, title, xyLabel, pngFPN)
                
            if self.modelPlot.rows.apply:
    
                self._PlotFeatureImportanceRows(featureArray, treeBasedImportanceArray, errorArray, 'treeBasedImportance', 'Mean impure. decr.')
    
            if self.multcompplot:
          
                self._MultCompPlotFeatureImportance(featureArray, treeBasedImportanceArray, errorArray, 'treeBasedImportance', 'Mean impure. decr.', multCompAxs)

        ''' Main function '''
        #Retrieve the model name and the model itself
        name,model = self.regrModel
        
        columns = [item for item in self.X_train_R.columns]
        
        #Fit the model
        model.fit(self.X_train_R, self.y_train_r)

        #Set the nr of feature (x-axis) items
        maxFeatures = min(self.modelling.featureImportance.reportMaxFeatures, len(columns))

        # Permutation importance
        PermImp()
        
        # Coefficient importance
        FeatureImp()
        
        # Treebased Importance
        TreeBasedImp()
        
    def _ManualFeatureSelector(self):
        '''
        '''

        IGNORESPREPROCESS
        # Reset columns
        columns = self.manualFeatureSelection.spectra

        # Create the dataframe for the sepctra
        spectraDF = self.spectraDF[ columns  ]
        
        X_train = self.X_train[ columns  ]

        self.manualFeatureSelectdRawBands =  columns
        
        self.manualFeatureSelectdDerivates = []
        
        # Create any derivative covariates requested
        
        if hasattr(self.manualFeatureSelection, 'derivatives'):
        
            if hasattr(self.manualFeatureSelection.derivatives, 'firstWaveLength'):
                
                for b in range(len(self.manualFeatureSelection.derivatives.firstWaveLength)):
        
                    bandL = [self.manualFeatureSelection.derivatives.firstWaveLength[b],
                             self.manualFeatureSelection.derivatives.lastWaveLength[b]]
    
                self.manualFeatureSelectdDerivates = bandL
    
                derviationBandDF = X_train[ bandL  ]
    
                bandFrame, bandColumn = self._SpectraDerivativeFromDf(derviationBandDF,bandL)
    
                frames = [X_train,bandFrame]
    
                #spectraDF = pd.concat(frames, axis=1)
    
                columns.extend(bandColumn)

        # reset self.spectraDF
        self.X_train = X_train

    def _VarianceSelector(self):
        '''
        '''

        threshold = self.generalFeatureSelection.varianceThreshold.threshold

        columns = [item for item in self.X_train.columns]
      
        #Initiate the scaler
        if self.generalFeatureSelection.varianceThreshold.scaler == 'None':
            
            X_train_scaled = Xscaled = self.X_train
            X_test_scaled = self.X_test
                       
        elif self.generalFeatureSelection.varianceThreshold.scaler == 'MinMaxScaler':

            scaler = MinMaxScaler()
            
            scaler.fit(self.X_train)
            
            #Scale the data as defined by the scaler
            Xscaled = scaler.transform(self.X_train)
            
            X_train_scaled = pd.DataFrame(scaler.transform(self.X_train), columns=columns)
            X_test_scaled = pd.DataFrame(scaler.transform(self.X_test), columns=columns)
            
        else:
            
            exitStr = 'EXITING - the scaler %s is not implemented for varianceThreshold ' %(self.generalFeatureSelection.varianceThreshold.scaler)

            exit (exitStr)
            
        if isinstance(threshold, float):
            select = VarianceThreshold(threshold=threshold)
   
        elif isinstance(threshold, int):
            select = VarianceThreshold(threshold=0.0000001)
            
        elif isinstance(threshold, str):
            select = VarianceThreshold(threshold=0.0000001)
            thresholdPercent = int(threshold[0:len(threshold)-1])
            threshold = int(round(len(columns)*thresholdPercent/100 ))
            
        #Initiate  VarianceThreshold
        #Fit the independent variables
        select.fit(Xscaled)

        #Get the selected features from get_support as a boolean list with True or False
        selectedFeatures = select.get_support()
        
        completeL = []
        
        #Create a list to hold discarded columns
        discardL = []

        #Create a list to hold retained columns
        retainL = []

        for sf in range(len(selectedFeatures)):

            completeL.append([columns[sf],select.variances_[sf]])
            
            if selectedFeatures[sf]:
                retainL.append([columns[sf],select.variances_[sf]])

            else:
                discardL.append([columns[sf],select.variances_[sf]])
               
        completeL.sort(key = lambda x: x[1]) 

        if isinstance(threshold, int) or isinstance(threshold, str) :
            
            discardL = completeL[0:threshold]
            
            retainL = completeL[threshold:len(retainL)]
            
        else:
            
            retainL.sort(key = lambda x: x[1])
            
            discardL.sort(key = lambda x: x[1])
                       
        if self.generalFeatureSelection.varianceThreshold.onlyShowVarianceList:
            
            print ('                covariate variance')
            print ('                band (variance)')
            printL = ['%s (%.3f)'%(i[0],i[1]) for i in completeL]

            for row in printL:
                print ('                ',row)
                
            print('') 
            
            sleep(2)
                
            exit('Select the varianceThrehsold{"threshold"} set varianceThrehsold{"showVarianceList"}) to true and rerun')
                
        if self.verbose:

            print ('            Selecting features using VarianceThreshold, threhold =',threshold)

            print ('                Scaling function MinMaxScaler:')
                
        self.generalFeatureSelectedD['method'] = 'varianceThreshold'
        self.generalFeatureSelectedD['threshold'] = self.generalFeatureSelection.varianceThreshold.threshold
        #self.generalFeatureSelectedD['scaler'] = self.generalFeatureSelection.scaler
        self.generalFeatureSelectedD['nCovariatesRemoved'] = len(discardL)

        #varianceSelectTxt = '%s covariate(s) removed with %s' %(len(discardL),'VarianceThreshold')
        
        generalFeatureSelectTxt = '%s covariate(s) removed with %s' %(len(discardL),'VarianceThreshold')

        #self.varianceSelectTxt = '%s: %s' %('VarianceThreshold',len(discardL))
        
        self.generalFeatureSelectTxt = '%s: %s' %('VarianceThreshold',len(discardL))

        if self.verbose:

            print ('            ',generalFeatureSelectTxt)

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

        retainL = [d[0] for d in retainL]
        
        discardL = [item[0] for item in discardL]
        
        PlotCoviariateSelection('Variance threshold', self.enhancementPlotLayout.varianceThreshold,  
                self.enhancementPlotLayout, self.preProcessFPND['varianceThreshold'],
                X_train_scaled, X_test_scaled, retainL, discardL, 
                self.Xcolumns, 'all', 'None',self.generalFeatureSelection.varianceThreshold.scaler)
        
        # Remake the X_train and X_test datasets
        self.X_train = self.X_train[ retainL ]
        
        self.X_test = self.X_test[ retainL ]
        
    def _ResetXY_T(self):
        
        self.X_train_T.reset_index(drop=True, inplace=True)
        
        self.X_test_T.reset_index(drop=True, inplace=True)
        
        self.y_train_t.reset_index(drop=True, inplace=True)
        
        self.y_test_t.reset_index(drop=True, inplace=True)
        
        # Remove all non-finite values       
        self.X_train_T = self.X_train_T[np.isfinite(self.y_train_t)] 
        
        self.y_train_t = self.y_train_t[np.isfinite(self.y_train_t)] 
        
        self.X_test_T = self.X_test_T[np.isfinite(self.y_test_t)] 

        self.y_test_t = self.y_test_t[np.isfinite(self.y_test_t)] 
        
    def _ResetXY_R(self):
        
        self.X_train_R.reset_index(drop=True, inplace=True)
        
        self.X_test_R.reset_index(drop=True, inplace=True)
        
        self.y_train_r.reset_index(drop=True, inplace=True)
        
        self.y_test_r.reset_index(drop=True, inplace=True)
        
        # Remove all non-finite values       
        self.X_train_R = self.X_train_R[np.isfinite(self.y_train_r)] 
        
        self.y_train_r = self.y_train_r[np.isfinite(self.y_train_r)] 
        
        self.X_test_R = self.X_test_R[np.isfinite(self.y_test_r)] 

        self.y_test_r = self.y_test_r[np.isfinite(self.y_test_r)]
        

    def _ResetTargetFeature(self):
        ''' Resets target feature for looping
        '''
        self.y_train_t = self.Y_train[self.targetFeature]
        
        self.y_test_t = self.Y_test[self.targetFeature]
                
        # Copy the X data
        self.X_train_T = deepcopy(self.X_train)
        
        self.X_test_T = deepcopy(self.X_test)

        # Remove all samples where the targetfeature is NaN
        self.X_train_T = self.X_train_T[~np.isnan(self.y_train_t)]
        
        self.y_train_t = self.y_train_t[~np.isnan(self.y_train_t)]
        
        self.X_test_T = self.X_test_T[~np.isnan(self.y_test_t)]
        
        self.y_test_t = self.y_test_t[~np.isnan(self.y_test_t)]
               
        self._ResetXY_T()
        
        self.X_columns_T = deepcopy(self.Xcolumns)
        
    def _PrepCovarSelection(self, nSelect):
        '''
        '''
        
        nfeatures = self.X_train_R.shape[1]

        if nSelect >= nfeatures:

            if self.verbose:

                infostr = '            SKIPPING specific selection: Number of input features (%s) less than or equal to minimumm output covariates to select (%s).' %(nfeatures, nSelect)

                print (infostr)

            return (False, False)
        
        columns = [item for item in self.X_train_R.columns]

        return (nfeatures, columns)   
    
    def _CleanUpCovarSelection(self, selector, selectorSymbolisation, retainL, discardL):
        '''
        '''
        
        self.specificFeatureSelectedD[self.targetFeature][self.regrModel[0]]['method'] = selector

        self.specificFeatureSelectedD[self.targetFeature][self.regrModel[0]]['nFeaturesRemoved'] = len( discardL)

        self.specificFeatureSelectionTxt = '- %s %s' %(len(discardL), selector)

        if self.verbose:
             
            print ('\n            specificFeatureSelection:')

            print ('                Regressor: %(m)s' %{'m':self.regrModel[0]})

            print ('                ',self.specificFeatureSelectionTxt)

            if self.verbose > 1:
    
                print ('                Selected features: %s' %(', '.join(retainL)))
        
        pngFPN = self.preProcessFPND['specificSelection'][self.targetFeature][self.regrModel[0]]

        label = '%s Selection' %(selector)
        
        PlotCoviariateSelection(label, selectorSymbolisation,  
                self.enhancementPlotLayout, pngFPN,
                self.X_train_R, self.X_test_R, retainL, discardL, 
                self.X_columns_R, self.paramD['targetFeatureSymbols'][self.targetFeature]['label'], self.regrModel[0])
   
        # reset the covariates
        self.X_train_R = pd.DataFrame(self.X_train_R, columns=retainL)
        
        self.X_test_R = pd.DataFrame(self.X_test_R, columns=retainL)
        
        self._ResetXY_R()
        
        # Reset columns
        valueL = []
        
        for item in retainL:
            if item in self.columns:
                
                valueL.append(self.columns[item])
                
        self.X_columns_R = dict(zip(retainL,valueL))
        
    def _CleanUpSpecificeAgglomeration(self, selector, selectorSymbolisation, retainL, discardL):
        '''
        '''

        self.specificClusteringdD[self.targetFeature]['nFeaturesRemoved'] = len( discardL)

        self.specificClusteringTxt = '- %s %s' %(len(discardL), selector)
        
        #agglomeratetxt = '%s input features clustered to %s covariates using  %s' %(len(columns),len(self.aggColumnL),self.globalFeatureSelectedD['method'])

        #self.agglomerateTxt = '%s clustered from %s to %s Feat´s' %(self.globalFeatureSelectedD['method'], len(columns),len(self.aggColumnL))
        #FIXATEXT

        if self.verbose:
              
            print ('\n            specificFeatureAgglomeration:')
                
            if self.verbose > 1:
    
                print ('                Selected features: %s' %(', '.join(retainL)))
                        
        pngFPN = self.preProcessFPND['specificClustering'][self.targetFeature]
                
        label = '%s Agglomeration' %(selector)

        PlotCoviariateSelection(label, selectorSymbolisation,  
                self.enhancementPlotLayout, pngFPN,
                self.X_train_T, self.X_test_T, retainL, discardL, 
                self.X_columns_T, self.paramD['targetFeatureSymbols'][self.targetFeature]['label'])
   
        # reset the covariates
        self.X_train_T = pd.DataFrame(self.X_train_T, columns=retainL)
        
        self.X_test_T = pd.DataFrame(self.X_test_T, columns=retainL)
        
        # Reset columns
        valueL = []
        
        for item in retainL:
            if item in self.columns:
                
                valueL.append(self.columns[item])
                
        self.X_columns_T = dict(zip(retainL,valueL))
                                     
    def _UnivariateSelector(self):
        '''
        '''

        nfeatures, columns = self._PrepCovarSelection(self.specificFeatureSelection.univariateSelection.SelectKBest.n_features)
        
        if not nfeatures:
            
            return
        
        # Apply SelectKBest
        select = SelectKBest(score_func=f_regression, k=self.specificFeatureSelection.univariateSelection.SelectKBest.n_features)
        
        # Select and fit the independent variables, return the selected array
        print ('xmax',self.X_train_R.max())
        print ('ymax',self.y_train_r.max())
        
        print ('xmin',self.X_train_R.min())
        print ('ymin',self.y_train_r.min())
        

        select.fit(self.X_train_R, self.y_train_r)
        
        

        # Note that the returned select.get_feature_names_out() is not a list
        retainL = select.get_feature_names_out()
        
        discardL = list( set(columns).symmetric_difference(retainL) )
        
        retainL.sort();  discardL.sort()
                
        # Save the results in a dictionary
        scores = select.scores_
        
        pvalues = select.pvalues_
        
        covars = select.feature_names_in_
        
        self.selectKBestResultD = {}
        
        for c, covar in enumerate(covars):
            
            self.selectKBestResultD[covar] = {'score': scores[c], 'pvalue': pvalues[c]}
         
        self._CleanUpCovarSelection('Univar SelKBest', self.enhancementPlotLayout.univariateSelection, retainL, discardL )
                               
    def _PermutationSelector(self):
        '''
        '''

        nfeatures, columns = self._PrepCovarSelection(self.specificFeatureSelection.permutationSelector.n_features_to_select)
        
        if not nfeatures:
            
            return
        
        #Retrieve the model name and the model itself
        model = self.regrModel[1]
        
        #Fit the model
        model.fit(self.X_train_R, self.y_train_r)

        permImportance = permutation_importance(model, self.X_test_R, self.y_test_r)

        permImportanceMean = permImportance.importances_mean

        sorted_idx = permImportanceMean.argsort()

        retainL = np.asarray(columns)[sorted_idx][::-1][0:self.specificFeatureSelection.permutationSelector.n_features_to_select].tolist()
        
        r = set(retainL)
        
        discardL = [x for x in columns if x not in r]
        
        retainL.sort(); discardL.sort()
        
        self._CleanUpCovarSelection('Permut Select', self.enhancementPlotLayout.permutationSelector, retainL, discardL)
        
    def _TreeBasedFeatureSelector(self):
        ''' NOTIMPLEMENTED
        '''
        
        pass
        
        # See https://scikit-learn.org/stable/modules/feature_selection.html

    def _RFESelector(self):
        '''
        '''

        nfeatures, columns = self._PrepCovarSelection(self.specificFeatureSelection.RFE.n_features_to_select)
        
        if not nfeatures:
            
            return

        step = self.specificFeatureSelection.RFE.step

        if self.verbose:

            if self.specificFeatureSelection.RFE.CV:

                print ('\n            RFECV feature selection')

            else:

                print ('\n            RFE feature selection')

        #Retrieve the model name and the model itself
        model = self.regrModel[1]

        if self.specificFeatureSelection.RFE.CV:

            select = RFECV(estimator=model, min_features_to_select=self.specificFeatureSelection.RFE.n_features_to_select, step=step, cv= self.specificFeatureSelection.RFE.CV)
            
            selector = 'RFECV'
      
        else:
            
            selector = 'RFE'
            
            select = RFE(estimator=model, n_features_to_select=self.specificFeatureSelection.RFE.n_features_to_select, step=step)
                
        select.fit(self.X_train_R, self.y_train_r)

        selectedFeatures = select.get_support()

        #Create a list to hold discarded columns
        retainL = []; discardL = []

        for sf in range(len(selectedFeatures)):
            if selectedFeatures[sf]:
                retainL.append(columns[sf])

            else:
                discardL.append(columns[sf])
                
        label = '%s Selection' %(selector)
        
        self._CleanUpCovarSelection(label, self.enhancementPlotLayout.RFE, retainL, discardL)       
                                            
    def _WardClustering(self, n_clusters):
        '''
        '''

        nfeatures = self.X_train_T.shape[1]

        if nfeatures < n_clusters:

            n_clusters = nfeatures
            
            return
        
        ward = FeatureAgglomeration(n_clusters=n_clusters)

        #fit the clusters
        ward.fit(self.X_train_T, self.y_train_t)

        self.clustering =  ward.labels_

        # Get a list of bands
        bandsL =  list(self.X_train_T)

        self.aggColumnL = []

        self.aggBandL = []
        
        discardL = []

        for m in range(len(ward.labels_)):

            indices = [bandsL[i] for i, x in enumerate(ward.labels_) if x == m]

            if(len(indices) == 0):

                break

            self.aggColumnL.append(indices[0])

            self.aggBandL.append( ', '.join(indices) )
            
            discardL.extend( indices[1:len(indices)])
            
        self.aggColumnL.sort()
        
        discardL.sort()
            
        self._CleanUpSpecificeAgglomeration('WardClustering', self.enhancementPlotLayout.wardClustering, self.aggColumnL, discardL )
                                           
    def _TuneWardClustering(self):
        ''' Determines the optimal nr of cluster
        '''
        nfeatures = self.X_train_T.shape[1]

        nClustersL = self.specificFeatureAgglomeration.wardClustering.tuneWardClustering.clusters

        nClustersL = [c for c in nClustersL if c < nfeatures]

        kfolds = self.specificFeatureAgglomeration.wardClustering.tuneWardClustering.kfolds

        cv = KFold(kfolds)  # cross-validation generator for model selection

        ridge = BayesianRidge()

        cachedir = tempfile.mkdtemp()

        mem = Memory(location=cachedir)

        ward = FeatureAgglomeration(n_clusters=4, memory=mem)

        clf = Pipeline([('ward', ward), ('ridge', ridge)])

        # Select the optimal number of parcels with grid search
        clf = GridSearchCV(clf, {'ward__n_clusters': nClustersL}, n_jobs=1, cv=cv)

        clf.fit(self.X_train_T, self.y_train_t)  # set the best parameters

        if self.verbose:

            print ('            Report for tuning Ward Clustering')

        #report the top three results
        self._ReportSearch(clf.cv_results_,3)

        #rerun with the best cluster agglomeration result
        tunedClusters = clf.best_params_['ward__n_clusters']

        if self.verbose:

            print ('                Tuned Ward clusters:', tunedClusters)

        return (tunedClusters)
    
    def _RemoveOutliers(self):
        """
        """
        
        if self.removeOutliers.contamination == 0:
            
            return
        
        def ExtractCovars(columnsX):
            '''
            '''
            
            extractTarget = False
            
            if 'target' in columnsX:
                
                targetIndex = columnsX.index('target')
                
                columnsX.pop(targetIndex)

                extractTarget = True
                
            xyTrainDF = pd.DataFrame(self.X_train_T, columns=columnsX)
            
            xyTrainDF.reset_index()
            
            xyTestDF = pd.DataFrame(self.X_test_T, columns=columnsX)
            
            xyTestDF.reset_index()
            
            if extractTarget:
            
                xyTrainDF['target'] = self.y_train_t

                xyTestDF['target'] = self.y_test_t
                
                columnsX.append('target')
            
            return (xyTrainDF, xyTestDF)
               
        def RemoveOutliers(Xtrain, Xtest,  columnsX):
            '''
            '''
            
            # extract the covariate columns as X
            X = Xtrain[columnsX]
    
            iniTrainSamples = X.shape[0]
    
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
            #yhat = outlierDetector.fit_predict(X)
            
            outlierFit = outlierDetector.fit(X)
            
            yhat = outlierFit.predict(X)
    
            # select all rows that are inliers           
            XtrainInliers = Xtrain[ yhat==1 ]
            
            # select all rows that are outliers
            XtrainOutliers = Xtrain[ yhat==-1 ]
            
            # Remove samples with outliers from the X and y dataset
            self.X_train_T = self.X_train_T[ yhat==1 ]
            
            self.y_train_t = self.y_train_t[ yhat==1 ]
                                    
            postTrainSamples = self.X_train_T.shape[0]
                        
            # Run the test data with the same detector
            # extract the covariate columns as X
            X = Xtest[columnsX]
            
            iniTestSamples = X.shape[0]
            
            yhat = outlierFit.predict(X)
            
            XtestInliers = X[ yhat==1 ]
            
            XtestOutliers = X[ yhat==-1 ]
            
            self.X_test_T = self.X_test_T[ yhat==1 ]
            
            self.y_test_t = self.y_test_t[ yhat==1 ]

            postTestSamples = self.X_test_T.shape[0]
            
            self.nTrainOutliers = iniTrainSamples - postTrainSamples
            self.nTestOutliers = iniTestSamples - postTestSamples
    
            self.outliersRemovedD['method'] = self.removeOutliers.detector
            self.outliersRemovedD['nOutliersRemoved'] = self.nTrainOutliers+self.nTestOutliers
    
            self.outlierTxt = '%s (%s) outliers removed ' %(self.nTrainOutliers,self.nTestOutliers )
    
            outlierTxt = '%s (%s) outliers removed' %(self.nTrainOutliers,self.nTestOutliers)
    
            if self.verbose:
    
                print ('            ',outlierTxt)
             
            if len(columnsX) == 2:
                
                # Use the orginal training data for defining the plot boundary between inliers and outliers
                X = Xtrain[columnsX]
                
                PlotOutlierDetect(self.enhancementPlotLayout, self.preProcessFPND['outliers'][self.targetFeature],
                                XtrainInliers, XtrainOutliers, XtestInliers, XtestOutliers,  
                                postTrainSamples, self.nTrainOutliers, postTestSamples, self.nTestOutliers,
                                self.removeOutliers.detector, columnsX, self.targetFeature, outlierFit, X)
                      
        ''' Main def'''
                
        columns = [item for item in self.X_train_T.columns]
            
        if len(self.removeOutliers.covarList) == 0:
            
            return 
          
        if self.removeOutliers.covarList[0] == '*':
        
            columnsX = [item for item in self.X_train_T.columns]
        
        else:
            
            columnsX = self.removeOutliers.covarList
            
        if self.removeOutliers.includeTarget:
            
            columnsX.append('target')
            
        for item in columnsX:
                
            if item != 'target' and item not in columns:
                    
                exitStr = 'EXITING - item %s missing in removeOutliers CovarL' %(item)
                
                exitStr += '\n    Available covars = %s' %(', '.join(columns) )
                
                exit(exitStr)
                            
        xyTrainDF, xyTestDF = ExtractCovars(columnsX)
                            
        RemoveOutliers(xyTrainDF, xyTestDF, columnsX)
        
        self._ResetXY_T()

    def _PcaPreprocess(self):
        """ See https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/tutorial/astronomy/dimensionality_reduction.html
            for faster (random) algorithm
        """ 
        
        Xcolumns = [item for item in self.X_train]
                
        if (len(Xcolumns) < self.spectraInfoEnhancement.decompose.pca.n_components):
            
            exit('EXITING - the number of surviving bands are less than the PCA components requested')
                
        # set the new covariate columns:
        columnsStr = []
        
        columnsInt = []
        
        for i in range(self.spectraInfoEnhancement.decompose.pca.n_components):
            if i > 100:
                x = 'pc-%s' %(i)
            elif i > 10:
                x = 'pc-0%s' %(i)
            else:
                x = 'pc-00%s' %(i) 
                
            columnsInt.append(i)
            columnsStr.append(x)
        
        self.columns = dict(zip(columnsStr, columnsInt))
        
        pca = PCA(n_components=self.spectraInfoEnhancement.decompose.pca.n_components)
                
        X_pc = pca.fit_transform(self.X_train)
         
        self.pcaComponents = pca.components_
        
        
        X_train_transformed = pca.fit_transform(self.X_train)
        
        X_test_transformed = pca.transform(self.X_test)
                
        PlotPCA(self.enhancementPlotLayout, self.preProcessFPND['decomposition'], 'pca', self.Xcolumns,
                                self.X_train, self.X_test, 
                                pd.DataFrame(data=X_train_transformed, columns=columnsStr), 
                                pd.DataFrame(data=X_test_transformed, columns=columnsStr))

        self.X_train = pd.DataFrame(data=X_train_transformed, columns=columnsStr)
        
        self.X_test = pd.DataFrame(data=X_test_transformed, columns=columnsStr)
               
        self.Xcolumns = dict(zip(columnsStr, columnsInt))

    def _RandomtuningParams(self,nFeatures):
        ''' Set hyper parameters for random tuning
        '''
        self.paramDist = {}

        self.HPtuningtxt = 'Random tuning'

        # specify parameters and distributions to sample from
        name, model = self.regrModel

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

        self.HPtuningtxt = 'Exhaustive tuning'

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

        
        search.fit(self.X_train_R, self.y_train_r)
        
        resultD = self._ReportSearch(search.cv_results_,self.params.hyperParameterTuning.n_best_report)

        self.tunedHyperParamsD[self.targetFeature][name] = resultD

        # Set the hyperParameters to the best result
        for key in resultD[1]['hyperParameters']:

            self.paramD['modelling']['regressionModels'][name]['hyperParams'][key] = resultD[1]['hyperParameters'][key]

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

       
        search.fit(self.X_train_R, self.y_train_r)
        
        resultD = self._ReportSearch(search.cv_results_,self.params.hyperParameterTuning.n_best_report)

        self.tunedHyperParamsD[self.targetFeature][name] = resultD

        # Set the hyperParameters to the best result
        for key in resultD[1]['hyperParameters']:

            self.paramD['modelling']['regressionModels'][name]['hyperParams'][key] = resultD[1]['hyperParameters'][key]

    def _ReportRegModelParams(self):
        '''
        '''

        print ('\n            %s hyper-parameters: %s' % (self.regrModel[0], self.regrModel[1]))
        '''
        for model in self.regressorModels:

            #Retrieve the model name and the model itself
            modelname,modelhyperparams = model

            print ('                name', modelname, modelhyperparams.get_params())
        '''

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
        paramD.pop('modelPlot')

        # Deep copy the parameters to self.soillineD
        self.modelPlotD = deepcopy(paramD)
                
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
        cmap = plt.get_cmap(self.modelPlot.colorramp)

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
            
            # Dict error [TGTODO check]
            if not 'abundances' in sample:
                
                continue

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
                                              
                    except:
                        
                        print('cannot box-cox transform')
                elif targetTransform.yeojohnson:
                    
                    pt = PowerTransformer()
                    
                    pt.fit(np.atleast_2d(self.abundanceDf[column]))
                     
                    print(pt.lambdas_)
                    
                    print(pt.transform( np.atleast_2d(self.abundanceDf[column]) ) )
                    
                    X = pt.transform( np.atleast_2d(self.abundanceDf[column]) )
                    
                    print (X)
                    
    
                            
                elif targetTransform.quantile:
                    
                    pass
                    
                    #X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
                    #qt = QuantileTransformer(n_quantiles=10, random_state=0)
                                                
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
        #self.varianceSelectTxt = None; 
        self.outlierTxt = None
        self.generalFeatureSelectTxt = None; 
        self.specificFeatureSelectionTxt = None; 
        self.agglomerateTxt = None

        # Use the wavelength as column headers
        columnsInt = self.jsonSpectraData['waveLength']

        # Convert the column headers to strings
        columnsStr = [str(c) for c in columnsInt]
        
        # create a headerDict to have the columns as both strings (key) and floats/integers (values)
        # Pandas DataFrame requires string, whereas pyplot requires numerical
        self.columns = dict(zip(columnsStr,columnsInt))
        
        self.originalColumns = dict(zip(columnsStr,columnsInt))
        
        n = 0

        # Loop over the spectra
        for sample in self.jsonSpectraData['spectra']:

            if n == 0:

                spectraA = np.asarray(sample['signalMean'])

            else:

                spectraA = np.vstack( (spectraA, np.asarray(sample['signalMean']) ) )

            n += 1

        self.spectraDF = pd.DataFrame(data=spectraA, columns=columnsStr)
        
    def _SetSubPlots(self):
        '''
        '''

        if self.modelPlot.rows.apply:

            self.nRegrModels = len(self.regressorModels)

            self.nTargetFeatures = len(self.targetFeatures)

            self.columnFig = {}

            self.columnAxs = {}
            
            

            if self.modelPlot.rows.targetFeatures.apply:

                self.targetFeaturePlotColumnD = {}

                for c, col in enumerate(self.modelPlot.rows.targetFeatures.columns):

                    self.targetFeaturePlotColumnD[col] = c

                self.targetFeaturesFigCols = len(self.modelPlot.rows.targetFeatures.columns)

                # Set the figure size
                xadd = self.modelPlot.rows.targetFeatures.figSize.xadd

                if  xadd == 0:

                    xadd = self.modelPlot.rows.subFigSize.xadd

                if self.modelPlot.rows.targetFeatures.figSize.x == 0:

                    figSizeX = self.modelPlot.rows.subFigSize.x * self.targetFeaturesFigCols + xadd

                else:

                    figSizeX = self.modelPlot.rows.targetFeatures.figSize.x

                yadd = self.modelPlot.rows.targetFeatures.figSize.yadd

                if  yadd == 0:

                    yadd = self.modelPlot.rows.subFigSize.yadd

                if self.modelPlot.rows.targetFeatures.figSize.y == 0:

                    figSizeY = self.modelPlot.rows.subFigSize.y * self.nTargetFeatures + yadd

                else:

                    figSizeY = self.modelPlot.rows.targetFeatures.figSize.y

                # Create column plots for individual targetFeatures, with rows showing different regressors
                for regrModel in self.regressorModels:

                    self.columnFig[regrModel[0]], self.columnAxs[regrModel[0]] = plt.subplots( self.nTargetFeatures, self.targetFeaturesFigCols, figsize=(figSizeX, figSizeY) )

                    if self.modelPlot.tightLayout:

                        self.columnFig[regrModel[0]].tight_layout()

                    # Set title
                    suptitle = "Regressor: %s; rows=target features; input features: %s;\n" %(regrModel[0], len(self.originalColumns))
                    suptitle += "%s; %s; %s; %s; \n" %(self.spectraInfoEnhancement.scatterCorrectiontxt, self.scalertxt, self.spectraInfoEnhancement.decompose.pcatxt, self.hyperParamtxt)
                    
                    # Set subplot wspace and hspace
                    if self.modelPlot.rows.regressionModels.hwspace.wspace:

                        self.columnFig[regrModel[0]].subplots_adjust(wspace=self.modelPlot.rows.regressionModels.hwspace.wspace)

                    if self.modelPlot.rows.regressionModels.hwspace.hspace:

                        self.columnFig[regrModel[0]].subplots_adjust(hspace=self.modelPlot.rows.regressionModels.hwspace.hspace)

                    #if self.varianceSelectTxt != None:
                    if self.generalFeatureSelectTxt != None:

                        suptitle += ', %s' %(self.generalFeatureSelectTxt)

                    if self.outlierTxt != None:

                        suptitle +=  ', %s' %(self.outlierTxt)

                    self.columnFig[regrModel[0]].suptitle(  suptitle )

                    for r,rows in enumerate(self.targetFeatures):

                        for c,cols in enumerate(self.modelPlot.rows.targetFeatures.columns):

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

            if self.modelPlot.rows.regressionModels.apply:

                self.regressionModelPlotColumnD = {}

                for c, col in enumerate(self.modelPlot.rows.regressionModels.columns):

                    self.regressionModelPlotColumnD[col] = c

                self.regressionModelFigCols = len(self.modelPlot.rows.regressionModels.columns)

                # Set the figure size

                xadd = self.modelPlot.rows.regressionModels.figSize.xadd

                if  xadd == 0:

                    xadd = self.modelPlot.rows.subFigSize.xadd

                if self.modelPlot.rows.regressionModels.figSize.x == 0:

                    figSizeX = self.modelPlot.rows.subFigSize.x * self.regressionModelFigCols + xadd

                else:

                    figSizeX = self.modelPlot.rows.regressionModels.figSize.x

                yadd = self.modelPlot.rows.regressionModels.figSize.yadd

                if  yadd == 0:

                    yadd = self.modelPlot.rows.subFigSize.yadd

                if self.modelPlot.rows.regressionModels.figSize.y == 0:

                    figSizeY = self.modelPlot.rows.subFigSize.y * self.nRegrModels + yadd

                else:

                    figSizeY = self.modelPlot.rows.regressionModels.figSize.x

                # Create column plots for individual regressionModels, with rows showing different regressors
                for targetFeature in self.targetFeatures:

                    self.columnFig[targetFeature], self.columnAxs[targetFeature] = plt.subplots( self.nRegrModels, self.regressionModelFigCols, figsize=(figSizeX, figSizeY))

                    # ERROR If only one regressionModle then r == NONE

                    if self.modelPlot.tightLayout:

                        self.columnFig[targetFeature].tight_layout()

                    # Set subplot wspace and hspace
                    if self.modelPlot.rows.targetFeatures.hwspace.wspace:

                        self.columnFig[targetFeature].subplots_adjust(wspace=self.modelPlot.rows.targetFeatures.hwspace.wspace)

                    if self.modelPlot.rows.targetFeatures.hwspace.hspace:

                        self.columnFig[targetFeature].subplots_adjust(hspace=self.modelPlot.rows.targetFeatures.hwspace.hspace)

                    label = self.paramD['targetFeatureSymbols'][targetFeature]['label']

                    suptitle = "Target: %s, %s (rows=regressors)\n" %(label, self.hyperParamtxt )

                    suptitle += '%s input features' %(len(self.originalColumns))

                    if self.generalFeatureSelectTxt != None:

                        suptitle += ', %s' %(self.generalFeatureSelectTxt)

                    if self.outlierTxt != None:

                        suptitle +=  ', %s' %(self.outlierTxt)

                    # Set suotitle
                    self.columnFig[targetFeature].suptitle( suptitle )

                    # Set subplot titles:
                    for r,rows in enumerate(self.regressorModels):

                        for c,cols in enumerate(self.modelPlot.rows.regressionModels.columns):

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
               
            if self.spectraInfoEnhancement.derivatives.apply:
    
                if self.spectraInfoEnhancement.derivatives.join:
                    
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
        
        self.imageFPND = {}; self.preProcessFPND = {}
        
        self.preProcessFPND['outliers'] = {}; 
        self.preProcessFPND['specificClustering'] = {}
        self.preProcessFPND['specificSelection'] = {}
        
        # Image files unrelated to targets and regressors 
        filterExtractFN = '%sfilterextract.png' %(prefix)
        
        self.filterExtractPlotFPN = os.path.join(modelimageFP,filterExtractFN)
        
        self.preProcessFPND['scatterCorrection'] = os.path.join(modelimageFP, '%s_scatter-correction.png' %(prefix) ) 
        self.preProcessFPND['standardisation'] = os.path.join(modelimageFP, '%s_standardisation.png' %(prefix) ) 
        self.preProcessFPND['derivatives'] = os.path.join(modelimageFP, '%s_derivatives.png' %(prefix) )
        self.preProcessFPND['decomposition'] = os.path.join(modelimageFP, '%s_decomposition.png' %(prefix) )
        self.preProcessFPND['varianceThreshold'] = os.path.join(modelimageFP, '%s_varaince-threshold-selector.png' %(prefix))
        
        # the picke files save the regressor models for later use
        self.trainTestPickleFPND = {}

        self.KfoldPickleFPND = {}
        
        # loop over targetfeatures
        for targetFeature in self.paramD['targetFeatures']:
            
            self.preProcessFPND['outliers'][targetFeature] = os.path.join(modelimageFP, '%s_%s_specific-outliers.png' %(prefix, targetFeature) )
          
            self.preProcessFPND['specificClustering'][targetFeature] = os.path.join(modelimageFP, '%s_%s_specfic-cluster.png' %(prefix, targetFeature) )
            
            self.preProcessFPND['specificSelection'][targetFeature] = {}
            
            self.imageFPND[targetFeature] = {}

            self.trainTestPickleFPND[targetFeature] = {}; self.KfoldPickleFPND[targetFeature] = {}

            for regmodel in self.paramD['modelling']['regressionModels']:
                                
                self.preProcessFPND['specificSelection'][targetFeature][regmodel] = os.path.join(modelimageFP, '%s_%s_%s_specific-select.png' %(prefix, targetFeature, regmodel) )
                
                trainTestPickleFN = '%s%s_%s_%s_trainTest.xsp'    %(prefix,'modelid',targetFeature, regmodel)

                KfoldPickleFN = '%s%s_%s_%s_Kfold.xsp'    %(prefix,'modelid',targetFeature, regmodel)

                self.trainTestPickleFPND[targetFeature][regmodel] = os.path.join(pickleFP, trainTestPickleFN)

                self.KfoldPickleFPND[targetFeature][regmodel] = os.path.join(pickleFP, KfoldPickleFN)

                self.imageFPND[targetFeature][regmodel] = {}

                if self.modelling.featureImportance.apply:

                    self.imageFPND[targetFeature][regmodel]['featureImportance'] = {}

                    imgFN = '%s%s_%s-model_permut-imp.png'    %(prefix,targetFeature, regmodel)

                    self.imageFPND[targetFeature][regmodel]['featureImportance']['permutationImportance'] = os.path.join(modelimageFP, imgFN)

                    imgFN = '%s%s_%s-model_feat-imp.png'    %(prefix,targetFeature, regmodel)

                    self.imageFPND[targetFeature][regmodel]['featureImportance']['regressionImportance'] = os.path.join(modelimageFP, imgFN)
                    
                    imgFN = '%s%s_%s-model_treebased-imp.png'    %(prefix,targetFeature, regmodel)

                    self.imageFPND[targetFeature][regmodel]['featureImportance']['treeBasedImportance'] = os.path.join(modelimageFP, imgFN)

                if self.modelling.modelTests.trainTest.apply:

                    imgFN = '%s%s_%s-model_tt-result.png'    %(prefix,targetFeature, regmodel)

                    self.imageFPND[targetFeature][regmodel]['trainTest'] = os.path.join(modelimageFP, imgFN)

                if self.modelling.modelTests.Kfold.apply:

                    imgFN = '%s%s_%s-model_kfold-result.png'    %(prefix,targetFeature, regmodel)

                    self.imageFPND[targetFeature][regmodel]['Kfold'] = os.path.join(modelimageFP, imgFN)

            # Set multi row-image file names, per targetfeature
            imgFN = '%s%s-multi-results.png'    %(prefix, targetFeature)

            self.imageFPND[targetFeature]['allmodels'] = os.path.join(modelimageFP, imgFN)

        for regmodel in self.paramD['modelling']['regressionModels']:

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
            
        if self.spectraInfoEnhancement.standardisation.apply:
            
            resultD['standardisation'] = True
            
        if self.spectraInfoEnhancement.decompose.pca.apply:
            
            resultD['pcaPreproc'] = True

        if self.removeOutliers.apply or self.generalFeatureSelection.apply:

            resultD['generalTweaks']= {}

            if self.removeOutliers.apply:

                resultD['generalTweaks']['removeOutliers'] = self.outliersRemovedD
                
            if self.generalFeatureSelection.apply:
                
                # either variance threshold or clustering - result saved in self.generalFeatureSelectedD
                resultD['generalTweaks']['generalFeatureSelection'] = self.generalFeatureSelectedD
                
                '''
                if self.generalFeatureSelection.varianceThreshold.apply:
    
                    resultD['generalTweaks']['varianceThreshold'] = self.varianceThresholdD
    
                if self.specificFeatureAgglomeration.apply:
    
                    resultD['generalTweaks']['featureAgglomeration'] = self.generalFeatureSelectedD
                '''
                
        if self.manualFeatureSelection.apply:

            resultD['manualFeatureSelection'] = True

        if self.specificFeatureSelection.apply:

            resultD['specificFeatureSelection'] = self.targetFeatureSelectedD

        if self.specificFeatureSelection.apply:

            resultD['modelFeatureSelection'] = self.specificFeatureSelectedD

        if self.modelling.featureImportance:

            resultD['featureImportance'] = self.modelFeatureImportanceD

        if self.modelling.hyperParameterTuning.apply:

            resultD['hyperParameterTuning'] = {}

            if self.modelling.hyperParameterTuning.randomTuning.apply:

                # Set the results from the hyperParameter Tuning
                resultD['hyperParameterTuning']['randomTuning'] = self.tunedHyperParamsD

            if self.modelling.hyperParameterTuning.exhaustiveTuning.apply:

                # Set the results from the hyperParameter Tuning
                resultD['hyperParameterTuning']['exhaustiveTuning'] = self.tunedHyperParamsD

        # Add the finally selected bands

        resultD['appliedModelingFeatures'] = self.finalFeatureLD

        # Add the final model results
        if self.modelling.modelTests.apply:

            resultD['modelResults'] = {}
            
            summaryD['modelResults'] = {}

            if self.modelling.modelTests.trainTest.apply:

                resultD['modelResults']['trainTest'] = self.trainTestResultD
                
                summaryD['modelResults']['trainTest'] = self.trainTestSummaryD
                
                for targetFeature in self.targetFeatures:
                
                    self.multCompSummaryD[targetFeature]['trainTest'] = self.trainTestSummaryD[targetFeature]
                
                    self.multCompSummaryD[targetFeature]['parameters'] = self.paramJsonFPN
                    
                    self.multCompSummaryD[targetFeature]['results'] = self.regrJsonFPN
                    
            if self.modelling.modelTests.Kfold.apply:

                resultD['modelResults']['Kfold'] = self.KfoldResultD
                
                summaryD['modelResults']['Kfold'] = self.KfoldSummaryD
                
                for targetFeature in self.targetFeatures:
                
                    self.multCompSummaryD[targetFeature]['Kfold'] = self.KfoldSummaryD[targetFeature]
                    
                    self.multCompSummaryD[targetFeature]['parameters'] = self.paramJsonFPN
                    
                    self.multCompSummaryD[targetFeature]['results'] = self.regrJsonFPN

        if self.verbose > 2:
            
            pp = pprint.PrettyPrinter(indent=2)
            pp.pprint(resultD)

        jsonF = open(self.regrJsonFPN, "w")
        
        json.dump(resultD, jsonF, indent = 2)

        jsonF = open(self.paramJsonFPN, "w")

        json.dump(self.paramD, jsonF, indent = 2)
        
        jsonF = open(self.summaryJsonFPN, "w")

        json.dump(summaryD, jsonF, indent = 2)

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
        
        # Creata a list for all images
        self.figLibL = []

        # Get the band data as self.spectraDF
        self._GetBandData()

        # Get and add the abundance data
        self._GetAbundanceData()

        self.hyperParamtxt = "hyper-param tuning: None"

        if self.modelling.hyperParameterTuning.apply:

            if self.modelling.hyperParameterTuning.exhaustiveTuning.apply:

                hyperParameterTuning = 'ExhaustiveTuning'

                self.tuningParamD = ReadModelJson(self.input.hyperParameterExhaustiveTuning)

                self.hyperParamtxt = "hyper-param tuning: grid search"

            elif self.modelling.hyperParameterTuning.randomTuning.apply:

                hyperParameterTuning = 'RandomTuning'

                self.tuningParamD = ReadModelJson(self.input.hyperParameterRandomTuning)

                self.hyperParamtxt = "hyper-param tuning: random"

            else:

                errorStr = 'Hyper parameter tuning requested, but no method assigned'

                exit(errorStr)

            self.hyperParams = Obj(self.tuningParamD )

        # Set the dictionaries to hold the model results
        self.trainTestResultD = {}; self.KfoldResultD  = {}; self.tunedHyperParamsD = {}
        self.generalFeatureSelectedD = {}; self.outliersRemovedD = {}; 
        self.targetFeatureSelectedD = {}
        self.specificFeatureSelectedD = {} 
        self.specificClusteringdD = {}
        self.modelFeatureImportanceD = {}
        self.finalFeatureLD = {}
        self.trainTestSummaryD = {}; self.KfoldSummaryD  = {};

        # Create the subDicts for all model + target related results
        for targetFeature in self.targetFeatures:

            self.tunedHyperParamsD[targetFeature] = {}; self.trainTestResultD[targetFeature] = {}
            self.KfoldResultD[targetFeature] = {}; self.specificFeatureSelectedD[targetFeature] = {}
            self.targetFeatureSelectedD[targetFeature] = {}; self.modelFeatureImportanceD[targetFeature] = {}
            self.finalFeatureLD[targetFeature] = {}
            self.trainTestSummaryD[targetFeature] = {}; self.KfoldSummaryD[targetFeature]  = {};
            self.specificClusteringdD[targetFeature] = {}

            for regModel in self.paramD['modelling']['regressionModels']:

                if self.paramD['modelling']['regressionModels'][regModel]['apply']:

                    self.trainTestResultD[targetFeature][regModel] = {}
                    self.KfoldResultD[targetFeature][regModel] = {}
                    self.specificFeatureSelectedD[targetFeature][regModel] = {}
                    self.modelFeatureImportanceD[targetFeature][regModel] = {}
                    self.finalFeatureLD[targetFeature][regModel] = {}
                    self.trainTestSummaryD[targetFeature][regModel] = {} 
                    self.KfoldSummaryD[targetFeature][regModel] = {}
                    
                    # Set the transformation to the output dict      
                    self.trainTestSummaryD[targetFeature]['transform'] = self.transformD[targetFeature]
                    
                    self.KfoldSummaryD[targetFeature]['transform'] = self.transformD[targetFeature]
                    
                    self.trainTestResultD[targetFeature]['transform'] = self.transformD[targetFeature]
 
                    self.KfoldResultD[targetFeature]['transform'] = self.transformD[targetFeature]
                        
                    if self.paramD['modelling']['hyperParameterTuning']['apply'] and self.tuningParamD[hyperParameterTuning][regModel]['apply']:

                        self.tunedHyperParamsD[targetFeature][regModel] = {}


        self.spectraInfoEnhancement.scatterCorrectiontxt, self.scalertxt, self.spectraInfoEnhancement.decompose.pcatxt, self.hyperParamtxt = 'NA','NA','NA','NA'
        
        self._SetSubPlots()
        
        # Filtering is applied separately for each spectrum and does not affect the distribution
        # between train and test datasets
        if self.spectraPreProcess.filtering.apply:
                
            filtertxt = self._FilterPrep()
                
        elif self.spectraPreProcess.multifiltering.apply:
                
            filtertxt = self._MultiFiltering()
            
        # Extract a new pandas DF 
        # The DF used in the loop must have all rows with NaN for the target feature removed        
        self._ExtractDataFrameX()
        
        self._ResetDataFramesXY()
        
        # Scatter correction: except for Multiplicative Scatter Correction (MSC),
        # the correction is strictly per spectrum and could be done prior to the split 
        # MSC reguires a Meanspectra - that is returned from the function
        self.spectraInfoEnhancement.scatterCorrectiontxt = 'scatter correction: None'
                 
        if self.spectraInfoEnhancement.apply:
             
            if self.spectraInfoEnhancement.scatterCorrection.apply:
                
                scatterCorrectiontxt, self.X_train, self.X_test, self.scattCcorrMeanSpectra = \
                    ScatterCorrection(self.X_train, self.X_test, 
                    self.spectraInfoEnhancement.scatterCorrection, self.enhancementPlotLayout,
                    self.preProcessFPND['scatterCorrection'])
                
                self.spectraInfoEnhancement.scatterCorrectiontxt = 'scatter correction: %s' %(scatterCorrectiontxt)
                
            # standardisation can do meancentring, z-score normalisation, paretoscaling or poissionscaling
            # the standardisation is defined from the training data and applied to the testdata
            # 2 vectors are required for the standardisation; mean and variance (scaling) 
            scaler = 'None'
            
            if self.spectraInfoEnhancement.standardisation.apply:
                  
                #scaler, scalerMean, ScalerScale = self._Standardisation()
                self.X_train, self.X_test, scaler, scalerMean, ScalerScale = Standardisation(self.X_train, self.X_test,
                                                self.spectraInfoEnhancement.standardisation, 
                                                self.enhancementPlotLayout, 
                                                self.preProcessFPND['standardisation'])
                
            if self.spectraInfoEnhancement.derivatives.apply:
                
                self.X_train, self.X_test, self.Xcolumns = Derivatives(self.X_train, 
                                self.X_test, self.spectraInfoEnhancement.derivatives.join, 
                                self.Xcolumns, self.enhancementPlotLayout, self.preProcessFPND['derivatives'])

            if self.spectraInfoEnhancement.decompose.apply:
                        
                if self.spectraInfoEnhancement.decompose.pca.apply:
                    
                    # PCA preprocess    
                    self.spectraInfoEnhancement.decompose.pcatxt = 'pca: None'
                    
                    self._PcaPreprocess()
                    
                    self.spectraInfoEnhancement.decompose.pcatxt = 'pca: %s comps' %(self.spectraInfoEnhancement.decompose.pca.n_components)
         
        # Any manual feature selection is applied to the original dataframe - i.e. affect all models the same
        if self.manualFeatureSelection.apply:

            self._ManualFeatureSelector()
                       
        if self.generalFeatureSelection.apply:

            if self.generalFeatureSelection.varianceThreshold.apply:
                
                self._VarianceSelector()
        
        # Loop over the target features to model
        for self.targetN, self.targetFeature in enumerate(self.targetFeatures):

            if self.verbose:

                infoStr = '\n    Target feature: %s' %(self.targetFeature)

            self._ResetTargetFeature()
            # RemoveOutliers is applied per target feature
            
            if self.removeOutliers.apply:
    
                self._RemoveOutliers()
                
            # Covariate (X) Agglomeration
            if self.specificFeatureAgglomeration.apply:

                if self.specificFeatureAgglomeration.wardClustering.apply:
                                            
                    if self.specificFeatureAgglomeration.wardClustering.tuneWardClustering.apply:
                        
                        n_clusters = self._TuneWardClustering()

                    else:

                        n_clusters = self.specificFeatureAgglomeration.wardClustering.n_clusters
                    
                    self._WardClustering(n_clusters)
                
            self._SetTargetFeatureSymbol()
            
            #Loop over the defined models
            for self.regrN, self.regrModel in enumerate(self.regressorModels):
                
                print ('        regressor:', self.regrModel[0])
                
                #RESET COVARS
                self._ResetRegressorXyDF()
                
                
                # Specific feature selection - max one applied in each model
                if  self.specificFeatureSelection.apply:
                            
                    if self.specificFeatureSelection.univariateSelection.apply:
                        
                        if self.specificFeatureSelection.univariateSelection.SelectKBest.apply:
                        
                            self._UnivariateSelector()
                                                  
                    elif self.specificFeatureSelection.permutationSelector.apply:
    
                        self._PermutationSelector()
    
                    elif self.specificFeatureSelection.RFE.apply:
    
                        if self.regrModel[0] in ['KnnRegr','MLP', 'Cubist']:
                                
                            self._PermutationSelector()
    
                        else:

                            self._RFESelector()
                                
                    elif self.specificFeatureSelection.treeBasedSelector.apply:
    
                        self._TreeBasedFeatureSelection()

                if self.modelling.featureImportance.apply:

                    self._FeatureImportance(multCompAxs)

                if self.modelling.hyperParameterTuning.apply:

                    if self.modelling.hyperParameterTuning.exhaustiveTuning.apply:

                        self._ExhaustiveTuning()

                    elif self.modelling.hyperParameterTuning.randomTuning.apply:

                        self._RandomTuning()

                    # Reset the regressor with the optimized hyperparameter tuning
                    #NOTDONE
                    # Set the regressor models to apply
                    self._RegModelSelectSet()

                if self.verbose > 2:

                    # Report the regressor model settings (hyper parameters)
                    self._ReportRegModelParams()

                if isinstance(self.y_test_r,pd.DataFrame):
                     
                    exit ('obs is a dataframe, must be a dataseries')

                columns = [item for item in self.X_train_R.columns]
                
                # unchanged columns from the start as lists
                if type(columns) is list:
                    
                    self.finalFeatureLD[self.targetFeature][self.regrModel[0]] = columns
                
                else: # otherwise not

                    self.finalFeatureLD[self.targetFeature][self.regrModel[0]] = columns.tolist()
                                
                if self.modelling.modelTests.apply:

                    if self.modelling.modelTests.trainTest.apply:

                        self._RegrModTrainTest(multCompAxs)

                    if self.modelling.modelTests.Kfold.apply:

                        self._RegrModKFold(multCompAxs)
                        
        if self.modelPlot.rows.screenShow:

            plt.show()

        if self.modelPlot.rows.savePng:

            if self.modelPlot.rows.targetFeatures.apply:

                for regModel in self.paramD['modelling']['regressionModels']:

                    if self.paramD['modelling']['regressionModels'][regModel]['apply']:

                        self.columnFig[regModel].savefig(self.imageFPND[regModel]['alltargets'])

            if self.modelPlot.rows.regressionModels.apply:

                for targetFeature in self.targetFeatures:

                    self.columnFig[targetFeature].savefig(self.imageFPND[targetFeature]['allmodels'])
                    
        for regModel in self.paramD['modelling']['regressionModels']:

            if self.paramD['modelling']['regressionModels'][regModel]['apply']:
        
                plt.close(fig=self.columnFig[regModel])
                
        for targetFeature in self.targetFeatures:
            
            if self.modelPlot.rows.regressionModels.apply:

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
    
    # Get the regression model symbols
    regressionModelSymbolsD = ReadAnyJson(iniParams['regressionmodelsymbols'])
    
    # Get the enhancement plot layout 
    enhancementPlotLayoutD = ReadAnyJson(iniParams['enhancementplotlayout'])
    
    # Get the plot model layout 
    modelPlotD = ReadAnyJson(iniParams['modelplot'])
    
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
                                                                          multiProjectComparisonD)

        multCompFig, multCompAxs, multCompPlotsColumns = SetMultCompPlots( multiProjectComparisonD,targetFeatureSymbolsD, len(jsonProcessObjectL) )

        multCompSummaryD = {}
        
        for targetFeature in multiProjectComparisonD['targetFeatures']:
            
            multCompSummaryD[targetFeature] = {}

    modelNr = 0    
    
    #Loop over all json files
    for jsonObj in jsonProcessObjectL:

        print ('    jsonObj:', jsonObj)

        paramD = ReadModelJson(jsonObj)
        
        # Setting of single and/or dual scatter correction
        if paramD['spectraInfoEnhancement']['apply']:
        
            if paramD['spectraInfoEnhancement']['scatterCorrection']['apply']:
                           
                if len(paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']) == 0:
                
                    paramD['spectraInfoEnhancement']['scatterCorrection']['apply'] = False
            
                elif len(paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']) == 1:
                    
                    paramD['spectraInfoEnhancement']['scatterCorrection']['singles'] = \
                        paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']
                        
                    paramD['spectraInfoEnhancement']['scatterCorrection']['duals'] = []
                    
                else:
                    
                    paramD['spectraInfoEnhancement']['scatterCorrection']['singles'] = []
          
                    paramD['spectraInfoEnhancement']['scatterCorrection']['duals'] = \
                        paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']
                                       
        # Add the targetFeatureSymbols
        paramD['targetFeatureSymbols'] = targetFeatureSymbolsD['targetFeatureSymbols']
        
        paramD['featureImportancePlot'] = targetFeatureSymbolsD['featureImportancePlot']
        
        # Add the regressionModelSymbols 
        paramD['regressionModelSymbols'] = regressionModelSymbolsD['regressionModelSymbols']
        
        # Add the targetFeatureSymbols
        paramD['featureImportancePlot'] = targetFeatureSymbolsD['featureImportancePlot']
        
        paramD['enhancementPlotLayout'] = enhancementPlotLayoutD['enhancementPlotLayout']
                
        # Add the plot model layout
        paramD['modelPlot'] = modelPlotD['modelPlot']

        paramD['multcompplot'] = False
        
        if multiProjectComparisonD['apply']:
            
            paramD['multcompplot'] = True
            
            paramD['modelNr'] = modelNr
            
            paramD['multCompPlotsColumns'] = multCompPlotsColumns
            
            # Replace the list of targetFeatures in paramD
            paramD['targetFeatures'] = multiProjectComparisonD['targetFeatures']
            
            # Replace the applied regressors, but not the hyper parameter definitions    
            for regressor in paramD["modelling"]['regressionModels']:
                               
                paramD["modelling"]['regressionModels'][regressor]['apply'] = multiProjectComparisonD["modelling"]['regressionModels'][regressor]['apply'] 
                
            # Replace all the processing steps boolean apply            
            processStepD = {}
            
            processStepD['spectraPreProcess'] = {}; processStepD['spectraInfoEnhancement'] = {}
            processStepD['manualFeatureSelection'] = {}; processStepD['generalFeatureSelection'] = {};
            processStepD['removeOutliers'] = {}; processStepD['specificFeatureAgglomeration'] = {};
            processStepD['specificFeatureSelection'] = {}; 
            
            processStepD['spectraPreProcess'] = ['filtering','multifiltering']
            processStepD['spectraInfoEnhancement'] = ['scatterCorrection','standardisation','derivatives','decompose']
            processStepD['manualFeatureSelection'] = []
   
            processStepD['generalFeatureSelection'] = ['varianceThreshold']
            processStepD['removeOutliers'] = []
            processStepD['specificFeatureAgglomeration'] = ['wardClustering'] 
            processStepD['specificFeatureSelection'] = ['univariateSelection','permutationSelector',
                                                        'RFE']
            
            for pkey in processStepD:
                                
                paramD[pkey]['apply'] = multiProjectComparisonD[pkey]['apply']
                
                for subp in processStepD[pkey]:
                    
                    paramD[pkey][subp]['apply'] = multiProjectComparisonD[pkey][subp]['apply']
            
            # Replace the feature importance reporting
            paramD['modelling']['featureImportance'] = multiProjectComparisonD['modelling']['featureImportance']
            
             
            # Replace the model test
            paramD['modelling']['modelTests'] = multiProjectComparisonD['modelling']['modelTests']                          
            
        # Setting of single and/or dual scatter correction
        if paramD['spectraInfoEnhancement']['apply']:
        
            if paramD['spectraInfoEnhancement']['scatterCorrection']['apply']:
                           
                if len(paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']) == 0:
                
                    paramD['spectraInfoEnhancement']['scatterCorrection']['apply'] = False
            
                elif len(paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']) == 1:
                    
                    paramD['spectraInfoEnhancement']['scatterCorrection']['singles'] = \
                        paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']
                        
                    paramD['spectraInfoEnhancement']['scatterCorrection']['duals'] = []
                    
                else:
                    
                    paramD['spectraInfoEnhancement']['scatterCorrection']['singles'] = []
          
                    paramD['spectraInfoEnhancement']['scatterCorrection']['duals'] = \
                        paramD['spectraInfoEnhancement']['scatterCorrection']['scaler']
                        
        # Get the target feature transform
        targetFeatureTransformD = ReadAnyJson(paramD['input']['targetfeaturetransforms'])
    
        # Add the targetFeatureTransforms
        paramD['targetFeatureTransform'] = targetFeatureTransformD['targetFeatureTransform']
        
        # Invoke the modeling
        mlModel = MachineLearningModel(paramD)

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
    #OutliersTest()

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
    
    rootJsonFPN = '/Users/thomasgumbricht/docs-local/OSSL2/projects_model/model_ossl_v4.json'
    
    iniParams = ReadAnyJson(rootJsonFPN)
            
if type( iniParams['projFN']) is list: 
          
        for proj in iniParams['projFN']:
            
            projParams = deepcopy(iniParams)
            
            projParams['projFN'] = proj
            
            SetupProcesses(projParams)
           
else:
        
    SetupProcesses(iniParams)
