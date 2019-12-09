#data wrangling
import pandas as pd
import numpy as np
#preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline


def cut_group(text):
    '''Reduce the categories for the \'cut\' field'''

    groups={'Good':'Fair',}
    return groups.get(text,text)

def clarity_group(text):
    '''Reduce the categories for the \'clarity\' field'''

    groups={'SI1':'SI','SI2':'SI','VS2':'VS','VS1':'VS',
       'VVS2':'VVS','VVS1':'VVS','IF1':'IF'}
    return groups.get(text,text)


def sumbmit_creator(y_pred,name):
    ''' Creates the submition file'''

    df = pd.DataFrame(y_pred)
    df = df.rename(columns={0:'price'})
    df['id']=df.index
    df=df[['id','price']]
    df.to_csv(name,index=False)

    return 'Csv file ready!'

def pipe_transformations(PCA_components):
    '''Operations for formating the data'''

    steps = [
    
    StandardScaler(),
    Normalizer(),
    PCA(n_components=PCA_components)

    ]

    pipe = make_pipeline(*steps)
    return pipe