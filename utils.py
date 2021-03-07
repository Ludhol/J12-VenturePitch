# This file contains functions used in the data analysis
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

def hasNumbers(inputString):
  return any(char.isdigit() for char in inputString)

def findCategoryColumns(df):
  categoryColumns = []
  for col in df.columns:
    for item in df[col]:
      if not pd.isnull(item) and type(item)==str and len(item.split(';')) > 1 and not hasNumbers(item):
        categoryColumns.append(col)
        break
  return categoryColumns

def findUniqueCategories(series):
  categories = []
  for item in series:
    if not pd.isnull(item):
      tempCategories = item.split(';')
      categories += [category for category in tempCategories if category not in categories]
  return categories

def categorySplitter(series):
  columns = findUniqueCategories(series)
  data = np.zeros((len(series), len(columns)))
  output = pd.DataFrame(data=data, columns=columns, index=series.index)
  for comp, item in zip(series.index, series):
    if not pd.isnull(item):
      for tech in item.split(';'):
        output.loc[comp, tech] = 1
  return output

def tsSplitter(ts):
  ts = ts.split(';')
  output = [float(val) if val != 'n/a' else np.nan for val in ts]
  if max(output) > 10**7:
    return pd.Series()
  return pd.Series(output)

def nameSplitter(name):
  names = name.split(' ')
  return names[0]

def yearSplitter(name):
  names = name.split(' ')
  years = names[1].strip('()').split(',')
  years = [int(year) for year in years]
  return years

def intervalMid(val):
  if type(val)==str:
    interval = val.split('-')
    if len(interval)==2:
      interval = [float(numb) for numb in interval]
      return sum(interval)/len(interval)
  return float(val)

def getOutliers(series, threshold = 4, verbose = True):
  norm = pd.DataFrame(series.dropna())
  norm['score'] = (norm - norm.mean())/norm.std()
  if threshold:
    norm = norm[abs(norm.score) >= threshold]
  if verbose:
    print('Outliers for threshold:', threshold)
  return round(norm)

def getNearestNeighbors(treated, control, nMatches=1, scaler=True):
  if scaler:
    if type(treated)==pd.Series:
      treated=treated.to_numpy()
      treated = treated.reshape(1, -1)
    scaler = StandardScaler()
    scaler.fit(treated)
    tArr = scaler.transform(treated)
    cArr = scaler.transform(control)

  nbrs = NearestNeighbors(n_neighbors=nMatches, algorithm = 'ball_tree').fit(cArr)
  distances, indices = nbrs.kneighbors(tArr)
  indices = indices.reshape(indices.shape[0]*indices.shape[1])
  return control.iloc[indices]

