import math

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#%matplotlib notebook
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
matplotlib.rc('text', usetex = True)
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
plt.rcParams.update(params)

import sklearn
from sklearn.linear_model import LinearRegression

from datetime import timedelta


# =============================================================================
# helper functions for data import
# =============================================================================

def load_trajectories_true(path,max_norm=False):
    df_leak_ts = pd.DataFrame()
    
    # load csv from data folder
    df_leak_ts_ = pd.read_csv(path,
                             delimiter=';',
                             parse_dates=['Timestamp'], 
                             index_col='Timestamp',
                              decimal=',')

    df_leak_ts = pd.concat([df_leak_ts, df_leak_ts_], axis=0).fillna(0)
        
            
    # get leak start and end dates
    df_leak_start = df_leak_ts.ne(0).idxmax().sort_values()
    df_leak_end = df_leak_ts.iloc[::-1].ne(0).idxmax()  
    
    # organize leaks in temporal order
    df_leak_ts = df_leak_ts[df_leak_start.sort_values().index]
    
    # normalize trajectories by max-leak-size for leak-type-assignment
    df_leak_max = df_leak_ts.max()
    df_leak_ts = df_leak_ts/df_leak_max
                
    # create df_leak_overview
    lst_index=['leak_type', 'leak_start', 'leak_fix', 'delta_t_expert']
    df_leak_overview = pd.DataFrame(index=lst_index,
                                    columns=df_leak_ts.columns)
    
    # determin leak-type based on first leak instance
    lst_leak_type = []
    for i, t_start in enumerate(df_leak_start):
        first_leak_value = df_leak_ts.loc[t_start, df_leak_start.index[i]]
    
        if first_leak_value>0.95:
            lst_leak_type.append('abrupt')
        if first_leak_value<0.05:
            lst_leak_type.append('incipient')
            
    df_leak_overview.loc['leak_type'] = lst_leak_type
        
    for leak in df_leak_overview:
        df_leak_overview.loc['leak_start', leak] = pd.to_datetime(df_leak_start[leak])
        df_leak_overview.loc['leak_fix', leak] = pd.to_datetime(df_leak_end[leak]) 
    
    # labels from PERFECT_TEAM solution
    dct_true_start = dict({'p523': '2019-01-15 23:00',
                           'p827': '2019-01-24 18:30',
                           'p280': '2019-02-10 13:05', #DMA C
                           'p653': '2019-03-03 13:10',
                           'p710': '2019-03-24 14:15',
                           'p514': '2019-04-02 20:40',
                           'p331': '2019-04-20 10:10',
                           'p193': '2019-05-19 10:40',
                           'p277': '2019-05-30 21:55', #DMA C
                           'p142': '2019-06-12 19:55',
                           'p680': '2019-07-10 08:45',
                           'p586': '2019-07-26 14:40',
                           'p721': '2019-08-02 03:00',
                           'p800': '2019-08-16 14:00',
                           'p123' :'2019-09-13 20:05',
                           'p455': '2019-10-03 14:00',
                           'p762': '2019-10-09 10:15',
                           'p426': '2019-10-25 13:25',
                           'p879': '2019-11-20 11:55'})
    
    # overwrite trajectory starts by given times
    for pipe in dct_true_start.keys():
        df_leak_overview.loc['leak_start', pipe] = pd.Timestamp(dct_true_start[pipe])
    
    # re-scale df_leak_ts if max-normalization not desired
    if max_norm==False:
        df_leak_ts = df_leak_ts*df_leak_max
    
    # expert labels submitted during the BattleDim
    dct_expert_annotations = dict({'p523': '2019-01-15 23:00',
                                   'p827': '2019-01-24 18:30',
                                   'p280': '2019-02-10 13:40', #DMA C
                                   'p653': '2019-03-10 12:00',
                                   'p710': '2019-03-24 14:20',
                                   'p514': '2019-04-02 20:40',
                                   'p331': '2019-04-20 10:10',
                                   'p277': '2019-05-20 00:00', #DMA C
                                   'p193': '2019-06-01 18:55',
                                   'p142': '2019-06-12 19:55',
                                   'p680': '2019-07-10 08:45',
                                   'p586': '2019-08-01 00:30',
                                   'p721': '2019-08-18 19:00',
                                   'p800': '2019-08-19 07:40',
                                   'p455': '2019-10-15 00:00',
                                   'p762': '2019-10-21 18:00',
                                   'p426': '2019-10-25 13:40',
                                   'p879': '2019-11-25 00:00'})

    for pipe in dct_expert_annotations.keys():
        t_annotation = pd.Timestamp(dct_expert_annotations[pipe])
        delta_t = t_annotation-df_leak_overview.loc['leak_start', pipe]
        df_leak_overview.loc['delta_t_expert', pipe] = delta_t
        df_leak_overview.loc['delta_v_expert', pipe] = df_leak_ts[pipe].cumsum().loc[:t_annotation][-2]/12  # from 1m³ per hour in 5min resolution
    
    df_leak_overview.loc['delta_v_expert', 'p123'] = df_leak_ts[pipe].cumsum()[-2]/12 # volume of missed annotation
    
    return df_leak_ts, df_leak_overview


def load_trajectories_detected(path,max_norm=False, roll_avg=1):
    df_leak_det = pd.read_csv(path,
                              delimiter=',',
                              decimal='.',
                              index_col='Timestamp')
    
    if roll_avg>1:
        df_leak_det = df_leak_det.rolling(roll_avg).mean().fillna(0)
    
    if max_norm==True:
        df_leak_det = abs(df_leak_det/abs(df_leak_det).max())
    
    return df_leak_det

    
# =============================================================================
# helper classes leakage identification
# =============================================================================

class Ref_node():
    def __init__(self,name):
        self.name = name
    def set_models(self, models):
        self._models_Reg = models

class State():
    def __init__(self, start, end, cor_start, cor_end, models_Reg=None):
        self._models_Reg = models_Reg
        self._start = start
        self._end = end
        self._cor_start = cor_start
        self._cor_end = cor_end
    def set_models(self, models):
        self._models_Reg = models

class SCADA_data():
    def __init__(self, pressures=None, flows=None, demands=None, levels=None):
        self.pressures = pressures
        self.flows = flows
        self.demands = demands
        self.levels = levels
    def load(self,path,sep=';',decimal=','):
        self.pressures = pd.read_csv(path +'Pressures'+'.csv',
                                           parse_dates = ['Timestamp'],
                                           dayfirst=True, sep=sep, decimal=decimal)
        self.pressures.index = pd.to_datetime(self.pressures['Timestamp'])
        self.pressures = self.pressures.drop('Timestamp', axis='columns')

        self.flows = pd.read_csv(path +'Flows'+'.csv',
                                       parse_dates = ['Timestamp'],
                                       dayfirst=True, sep=sep, decimal=decimal)
        self.flows.index = pd.to_datetime(self.flows['Timestamp'])
        self.flows = self.flows.drop('Timestamp', axis='columns')

        self.demands = pd.read_csv(path +'Demands'+'.csv',
                                         parse_dates = ['Timestamp'],
                                         dayfirst=True, sep=sep, decimal=decimal)
        self.demands.index = pd.to_datetime(self.demands['Timestamp'])
        self.demands = self.demands.drop('Timestamp', axis='columns')

        self.levels = pd.read_csv(path +'Levels'+'.csv',
                                        parse_dates = ['Timestamp'],
                                        dayfirst=True, sep=sep, decimal=decimal)
        self.levels.index = pd.to_datetime(self.levels['Timestamp'])
        self.levels = self.levels.drop('Timestamp', axis='columns')
    
# =============================================================================
# helper functions cp-detection
# =============================================================================

def cusum(df,direction='p', ar_mean=[0], ar_K=[0]):
    """Tabular CUSUM per Montgomery,D. 1996 "Introduction to Statistical Process Control" p318 
    x    :  df to analyze
    mean :  expected process mean for each pipe 
            (if int/float: same for all pipes / if np.array: individual means)
    K    :  reference value, allowance, slack value for each pipe 
            (if int/float: same for all pipes / if np.array: individual Ks)
    ---        
    df_cs:  pd.DataFrame containing cusum-values for each df column
    """
    
    if (len(ar_mean)==1) | (len(ar_K)==1):
        ar_mean = np.full(df.shape[1], ar_mean[0])
        ar_K = np.full(df.shape[1], ar_k[0])
             
    cumsum=np.zeros(df.shape)
    
    if direction=='p':
        for i in range(1,df.shape[0]):
            cumsum[i,:] = [max(0,j) for j in df.iloc[i,:]-ar_mean+cumsum[i-1,:]-ar_K]
    elif direction=='n':
        for i in range(1,df.shape[0]):
            cumsum[i] = [max(0,j) for j in -df.iloc[i,:]+ar_mean+cumsum[i-1,:]-ar_K]

    df_cs = pd.DataFrame(cumsum, columns=df.columns, index=df.index)
    return df_cs


def zscore(df, win):
    """calcualte rolling z_score for leak trajectories
    df :    dfs containing leak trajectories (model reconstruction errors)
    win :   window for error statistics calculation
    
    df_z :  pd.DataFrame containing normalized trajectories
    """ 
    start_dates = df.ne(0).idxmax()
    z_base = np.zeros(shape=(win,df.shape[1]))
    
    for i, pipe in enumerate(df):
        start = start_dates[pipe]
        stop = start_dates[pipe]+pd.Timedelta(win, unit='Min')*5
        z_base[:,i] = df[pipe].loc[start:stop].iloc[:-1]

    m = z_base.mean(axis=0)
    sigma = z_base.std(axis=0)
    z = (df-m)/sigma
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_z = pd.DataFrame(z, columns=df.columns, index=df.index)   
    # 
    return df_z, sigma, m