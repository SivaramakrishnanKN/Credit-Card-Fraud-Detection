import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as e_dist
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA 
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from MulticoreTSNE import MulticoreTSNE as MTSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import copy
import umap

df = pd.read_csv("creditcard.csv", dtype = 'float32')

df = df.sample(frac=1, random_state = 3)
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:5000]

#new_df = df.loc[6000:7000]

scaler = StandardScaler()
scaler1 = MinMaxScaler()
# Shuffle dataframe rows and Scale Time and Amount
new_df = pd.concat([fraud_df, non_fraud_df])
new_df = new_df.sample(frac=1, random_state=42)

new_df[['Time', 'Amount']] = scaler1.fit_transform(new_df[['Time', 'Amount']])

print(len(new_df[new_df['Class']==1]))

prox_mat = e_dist(new_df, new_df)# Proximity matrix

eps = 4
min_pts = 4

core = []
border = []
noise = []

def in_list(val,lists):
    for i in lists:
        if i==val:
            return 1
    return 0

def neighbor_points(val):
    count = 0
    for i in prox_mat[val]:
        if i<=eps:
            count+=1     
    return count

def core_neighbor(val):
    for i in core:
        if prox_mat[i][val]<=eps:
            return 1
    return 0

def cluster_core(valu):
  l = []
  temp_clust = []
  temp_clust.append(valu)
  l.append(valu)
  unvisited.remove(valu)
  while len(l):
    val = l.pop()
    for i in unvisited:
      if prox_mat[val][i]<=eps and in_list(i,unvisited) and i!=val:
        temp_clust.append(i)
        unvisited.remove(i)
        l.append(i)
        #visited.append(i)
        #cluster_core(i, temp_clust)
  return temp_clust  

# Core Points
t0 = time.time()
for i in range(len(prox_mat)):
    if neighbor_points(i)>=min_pts:
        core.append(i)
t1 = time.time()
print("T-SNE took {:.2} s".format(t1 - t0))
core = list(dict.fromkeys(core))


t0 = time.time()
# Border Points
for i in range(len(prox_mat)):
    if in_list(i,core)==0 and core_neighbor(i)==1:
        border.append(i)
t1 = time.time()
print("T-SNE took {:.2} s".format(t1 - t0))
border = list(dict.fromkeys(border))



t0 = time.time()
# Noise points
for i in range(len(prox_mat)):
    if in_list(i,core)==0 and in_list(i,border)==0:
        noise.append(i)
t1 = time.time()
print("T-SNE took {:.2} s".format(t1 - t0))
noise = list(dict.fromkeys(noise))

# Removing noise points
# =============================================================================
# for i in noise:
#     prox_mat=np.delete(prox_mat,(i),axis=0)
#     prox_mat=np.delete(prox_mat,(i),axis=1)
# 
# =============================================================================

features = list(new_df.columns.values)
features.remove('Class')
x = new_df.loc[:, features].values
# Separating out the target
y = new_df.loc[:,['Class']].values
# Standardizing the features

y_pred = []
for i in range(len(new_df)):
  if in_list(i, noise):
    y_pred.append(1)
  else:
    y_pred.append(0)

corr = 0
incorr = 0
for i in range(len(new_df)) :
  if y_pred[i] == y[i]:
    corr +=1
  else:
    incorr +=1
new_df['Predict'] = y_pred

print(corr/(corr+incorr)*100,'%')

##======================================== Clustering core points =================================
#clusters = []
#unvisited = copy.deepcopy(core)
#visited = []
#
#while len(unvisited)>0:
#    for i in unvisited:
#      clusters.append(cluster_core(i))
#      
#      
##   adding border points to the respective clusters         
#for i in border:
#    min = 10000
#    for j in core:
#        dist = prox_mat[i,j]
#        if dist<min:
#            min = dist
#            index = j  
#    print(index)    
#    for j in clusters:
#        for k in j:
#            if k==index:
#                j.append(i)
#                break

# ===================================== Graph =================================================


    



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
groups = ['Fraud', 'Non Fraud']
for target, color, group in zip(targets,colors, groups):
    indicesToKeep = finalDf['Class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50
               , label = group)
ax.legend(loc=2)
ax.grid()
plt.show()

# ================================= Plotting with PCA =======================================
pca = PCA(n_components=2)
principalComponents1 = pca.fit_transform(x)
principalDf1 = pd.DataFrame(data = principalComponents1
             , columns = ['principal component 1', 'principal component 2'])
principalDf1['Class'] = y_pred
finalDf1 = principalDf1
fig1 = plt.figure(figsize = (8,8))
ax1 = fig1.add_subplot(1,1,1) 
ax1.set_xlabel('Principal Component 1', fontsize = 15)
ax1.set_ylabel('Principal Component 2', fontsize = 15)
ax1.set_title('Predicted', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
groups = ['Non Fraud', 'Fraud']
for target, color, group in zip(targets,colors, groups):
    indicesToKeep = finalDf1['Class'] == target
    ax1.scatter(finalDf1.loc[indicesToKeep, 'principal component 1']
               , finalDf1.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50
               , label = group)
ax1.legend(loc=2)
ax1.grid()
plt.show()
#
## ================== Plotting with Uniform Manifold Approximation and Projection =======================================
#t0 = time.time()
#umap_data = umap.UMAP(n_neighbors=150, min_dist=0.3, n_components=2)
#principalComponents = umap_data.fit_transform(x)
#principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['principal component 1', 'principal component 2'])
#principalDf['Class'] = y
#finalDf = principalDf
#fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1) 
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('2 component UMAP', fontsize = 20)
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
## ================================= Plotting with TSNE =======================================
#t0 = time.time()
##tsne = TSNE(n_components = 2)
#tsne = MTSNE(n_components = 2, n_jobs = 4)
#principalComponents = tsne.fit_transform(x)
#principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['principal component 1', 'principal component 2'])
#principalDf['Class'] = y_pred
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
## ================================= Plotting with Truncated SVD =======================================
#pca = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42)
#principalComponents = pca.fit_transform(x)
#principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['principal component 1', 'principal component 2'])
#principalDf['Class'] = y_pred
#finalDf = principalDf
#fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1) 
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('2 component TruncatedSVD', fontsize = 20)
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

## ================================= Plotting with ICA =======================================
#ICA = FastICA(n_components=2, random_state=12)
#principalComponents = ICA.fit_transform(x)
#principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['principal component 1', 'principal component 2'])
#principalDf['Class'] = y
#finalDf = principalDf
#fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1) 
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('2 component ICA', fontsize = 20)
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
  
## ================================= Plotting with PCA =======================================
#pca = PCA(n_components=2)
#principalComponents = pca.fit_transform(x)
#principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['principal component 1', 'principal component 2'])
#principalDf['Class'] = y
#finalDf = principalDf
#fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1) 
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('2 component PCA', fontsize = 20)
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
# =============================================================================
# for i in noise:
#     min = 100
#     for j in core:
#       dist = prox_mat[i,j]
#       if dist<min:
#         min = dist
#         index = j
#     for j in clusters:
#       for k in j:
#         if k==index:
#           j.append(i)
#           break
#       break    
#    
# =============================================================================
