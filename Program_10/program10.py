import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,classification_report
data=load_breast_cancer()
X=data.data
y=data.target
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
Kmeans=KMeans(n_clusters=2,random_state=42)
y_Kmeans=Kmeans.fit_predict(X_scaled)
pca=PCA(n_components=2)
x_pca=pca.fit_transform(X_scaled)
df=pd.DataFrame(x_pca,columns=['PC1','PC2'])
df['Cluster']=y_Kmeans
df['True label']=y
plt.figure(figsize=(8,6))
sns.scatterplot(data=df,x='PC1',y='PC2',hue='Cluster',palette='Set1',s=100,edgecolor='black',alpha=0.7)
plt.title('KMeans Clustering of Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(data=df,x='PC1',y='PC2',hue='Cluster',palette='Set1',s=100,edgecolor='black',alpha=0.7)
centers=pca.transform(Kmeans.cluster_centers_)
plt.scatter(centers[:,0],centers[:,1],s=200,c='red',marker='x',label='centriod')
plt.title('KMeans Clustering of Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title="Cluster")
plt.grid(True)
plt.show()
