import copy
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances as e_dist
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as MTSNE

#========== Importing Dataset =================================================================================
df = pd.read_csv("creditcard.csv", dtype = 'float32')

df = df.sample(frac=1, random_state = 32)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:5000]

new_df = pd.concat([fraud_df, non_fraud_df])

scaler = StandardScaler()
# Shuffle dataframe rows and Scale Time and Amount
new_df = new_df.sample(frac=1, random_state=2)
new_df[['Time', 'Amount']] = scaler.fit_transform(new_df[['Time', 'Amount']])

prox_mat = e_dist(new_df, new_df)
#new_df = [0, 1, 2, 3]
#prox_mat = [[0, 1, 2, 3],
#            [1, 0, 1, 4],
#            [2, 1, 0, 3],
#            [3, 4, 3, 0]]
#prox_mat = np.asarray(prox_mat)

#====================================== Variables =================================================================================
k = 450

k_distance = []
k_neighbours = []
lof = []
rd = []
lrd = []

#======================================== Function Definitions =======================================================================
def get_dist(i):
  temp = copy.deepcopy(prox_mat[i])
  temp = np.sort(temp)
  return temp[k]

def in_list(val,lists):
    for i in lists:
        if i==val:
            return 1
    return 0

def get_neighbours(i):
  temp = []
  for j in range(len(prox_mat[i])):
    if prox_mat[i,j]<=k_distance[i] and i!=j:
      temp.append(j)
  k_neighbours.append(temp)

def reachability_dist(a,b):
  return max(k_distance[b], prox_mat[a, b])

def local_reachability_density(a):
  s = 0
  for i in k_neighbours[a]:
    s+=rd[a][i]
  return len(k_neighbours[a])/s

def local_outlier_factor(a):
  s = 0
  for i in k_neighbours[a]:
    s+=lrd[i]
  s = s/len(k_neighbours[a])
  s = s/lrd[a]
  return s
#============================================= Creating K_distance, K_neighbours ==========================================================  

for i in range(len(new_df)):
  k_distance.append(get_dist(i))
  
for i in range(len(new_df)):
  get_neighbours(i)

for i in range(len(new_df)):
  temp = []
  for j in range(len(new_df)):
    temp.append(reachability_dist(i, j))
  rd.append(temp)

for i in range(len(new_df)):
  lrd.append(local_reachability_density(i))

for i in range(len(new_df)):
  lof.append(local_outlier_factor(i))

ll = range(len(lof))
lol = np.array([lof,ll, new_df['Class']])
lol = lol.transpose()
lol = lol[np.argsort(lol[:, 0])]


# ========================================== Printing Accuracy ===============================================================================
thresh = 1.5
cor = 0
incor = 0
thresh_ind = len(lol)

features = list(new_df.columns.values)
features.remove('Class')
x = new_df.loc[:, features].values
# Separating out the target
y = new_df.loc[:,['Class']].values
# Standardizing the features


for i in range(len(lol)):
  if lol[i][0] >= thresh:
    thresh_ind = i
    break


for i in range(len(lol)):
  if lol[i][0] < thresh:
    if lol[i][2]==0:
      cor = cor + 1
    else:
      incor = incor + 1
  else:
    if lol[i][2]==1:
      cor = cor + 1
    else:
      incor = incor + 1

print(cor/(cor+incor))

y_pred = []
for i in range(len(new_df)):
  if i<thresh_ind:
    y_pred.append(1)
  else:
    y_pred.append(0)


# ====================================== Graph =========================================================


# ================================= Plotting with PCA =======================================
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf['Class'] = y
finalDf = principalDf
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Original', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

# ================================= Plotting with PCA =======================================
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf['Class'] = y_pred
finalDf = principalDf
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Predicted', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

# ================================= Plotting with TSNE =======================================
#t0 = time.time()
##tsne = TSNE(n_components = 2)
#tsne = MTSNE(n_components = 2, n_jobs = 4)
#principalComponents = tsne.fit_transform(x)
#principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['principal component 1', 'principal component 2'])
#principalDf['Class'] = y
#finalDf = principalDf
#fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1) 
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('2 component TSNE', fontsize = 20)
#targets = [0, 1]
#colors = ['r', 'g']
#for target, color in zip(targets,colors):
#    indicesToKeep = finalDf['Class'] == target
#    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#               , finalDf.loc[indicesToKeep, 'principal component 2']
#               , c = color
#               , s = 50)
#ax.legend('Non Fraud', 'Fraud')
#ax.grid()
#plt.show()
#t1 = time.time()
#print("TSNE took {:.2} s".format(t1 - t0))
#
##================================ Visualizing Predictions ==============================================
#
#n_df = []
#for i in range(len(lol)):
#  temp = np.array(new_df.iloc[int(lol[0][1])])
#  if i<thresh_ind:
#    temp[30] = 0
#  else:
#    temp[30] = 1
#  n_df.append(temp)
#dff = pd.DataFrame(n_df, dtype = 'float32')
#dff.columns = ['0','1','2','3','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','Class']
#features = list(dff.columns.values)
#
#x = dff.loc[:, features].values
## Separating out the target
#y = dff.loc[:,['Class']].values
## Standardizing the features
#x = StandardScaler().fit_transform(x)
#
## ================================= Plotting with TSNE =======================================
#t0 = time.time()
##tsne = TSNE(n_components = 2)
#tsne = MTSNE(n_components = 2, n_jobs = 4)
#principalComponents = tsne.fit_transform(x)
#principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['principal component 1', 'principal component 2'])
#principalDf['Class'] = y
#finalDf = principalDf
#fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1) 
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('2 component TSNE', fontsize = 20)
#targets = [0, 1]
#colors = ['r', 'g']
#for target, color in zip(targets,colors):
#    indicesToKeep = finalDf['Class'] == target
#    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#               , finalDf.loc[indicesToKeep, 'principal component 2']
#               , c = color
#               , s = 50)
#ax.legend(targets)
#ax.grid()
#plt.show()
#t1 = time.time()
#print("PCA took {:.2} s".format(t1 - t0))
#
