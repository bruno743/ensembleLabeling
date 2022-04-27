from itertools import count
import warnings
import os.path
import numpy as np
import pandas as pd

from sklearn.preprocessing import minmax_scale
from .regression_model import _TrainingModels
from .rotulator_model import _range_delimitation
from .rotulate import _label

warnings.filterwarnings("ignore")

def _call_predictions(X, Y, XNormal, kernel, c, gamma):

    models =  _TrainingModels(XNormal, 10, kernel, c, gamma)
    predictions = models.predictions
    
    yy = pd.DataFrame(columns=['Atributo', 'Actual', 'Normalizado', 'Predicted', 'Cluster', 'Erro'])

    count = 0
    for attr in XNormal.columns:
        if attr == 'target': break
        y_ = pd.DataFrame(columns=['Atributo', 'Actual', 'Normalizado', 'Predicted', 'Cluster', 'Erro'])
        y_['Actual'] = [aux[count] for aux in X]
        y_['Normalizado'] = XNormal[attr].values
        y_['Predicted'] = predictions[(predictions['Atributo']==attr)].sort_values(by='index')['predict'].values
        y_['Cluster'] = Y
        y_ = y_.assign(Erro=lambda x: abs(x.Normalizado-x.Predicted))
        y_ = y_.assign(Atributo=attr)
        yy = pd.concat([yy, y_])
        count += 1

    return yy

def _poly_apro(results):
	polynomials = {}
	for attr, values in results.groupby(['Atributo']):
		d = {}
		for clt, data in values.groupby(['Cluster']):
			if data.shape[0] > 1:
				d[clt] = list(np.polyfit(data['Actual'].to_numpy().astype(float), data['ErroMedio'].to_numpy().astype(float), 2))
				polynomials[attr] = d

	return polynomials

def first_stage(df, dfN, X, Y, columnNames):
    kernel = 'linear'
    c = 100
    gama = 'auto'

    yy = _call_predictions(X, Y, dfN, kernel, c, gama)
    yy = yy.rename(columns={'Cluster' : 'target'})

    return yy, len(np.unique(yy['target']))

def final_stage(df, dfN, X, Y, columnNames, baseInformation):
    curves_diff = 0.8
    acceptable_error = 0.2

    baseInformation = baseInformation.rename(columns={'target' : 'Cluster'})

    errorByValue = (baseInformation.groupby(['Atributo', 'Cluster', 'Actual'])
        .agg({'Erro': np.average})
        .reset_index().astype({'Actual' : 'float64', 'Erro': 'float64'}))
	
    errorByValue = errorByValue.rename(columns={'Erro' : 'ErroMedio'})

    attrRangeByGroup = (baseInformation.groupby(['Atributo', 'Cluster']).agg({'Actual': [np.min, np.max]})
		.reset_index()
		.astype({'Actual': 'float64'}))
	
    attrRangeByGroup['minValue'] = attrRangeByGroup['Actual']['amin']
	
    attrRangeByGroup['maxValue'] = attrRangeByGroup['Actual']['amax']
	
    attrRangeByGroup = attrRangeByGroup.drop(['Actual'], axis=1)
	
    polynomials = _poly_apro(errorByValue)
	
    relevanteRanges = _range_delimitation(attrRangeByGroup, polynomials, curves_diff)

    ranged_attr, results, labels, rotulation_process = _label(relevanteRanges, acceptable_error, df)

    '''print(f'\nra:\n{ranged_attr}\n')
    print(f'\nr:\n{results}\n')
    print(f'\nl:\n{labels}\n')
    print(f'\nrp:\n{rotulation_process}\n')'''
    
    L = []
    for index, group in labels.drop('Precision', axis=1).groupby('Cluster'):
            L.append([(g[1], g[2], g[3]) for g in group.values])
    
    return L, results.drop('Cluster', axis=1).values