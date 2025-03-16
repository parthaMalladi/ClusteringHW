#INSTRUCTIONS:
##Order of running code files: recipe_cleanup.py-->graph_building.py-->recipe_clustering.py-->recipe_classification.py-->recipe_recommendation.py
#Performs t-SNE clustering for:Cuisines based on ingredients; Cuisines based on flavors and Generates interactive Bokeh plots
#Make sure to update the path of data frames
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

def convertToNumerical(data: pd.DataFrame) -> pd.DataFrame:
    # make a copy
    df = data.copy()

    # convert all boolean columns to numerical
    df[df.select_dtypes(include=['bool']).columns] = df.select_dtypes(include=['bool']).astype(int)

    # sort into numerical and categorical columns
    numerical = []
    categorical = []

    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            numerical.append(col)
        else:
            categorical.append(col)

    # convert all categorical columns to numerical
    le = LabelEncoder()
    for col in categorical:
        df[col] = le.fit_transform(df[col])

    return df

def DensityClustering(data, eps, minSamples):
    dbscan = DBSCAN(eps=eps, min_samples=minSamples)
    dbscan_labels = dbscan.fit_predict(data)

    # Reduce dimensions with PCA for visualization
    pca = PCA(n_components=0.95)
    reduced = pca.fit_transform(data)

    # Plot DBSCAN clusters
    plt.figure(figsize=(8, 5))
    unique_labels = set(dbscan_labels)

    bright_colors = [
        'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'pink', 'purple', 'lightblue',
        'lime', 'violet', 'turquoise', 'indigo', 'gold', 'salmon', 'coral', 'chartreuse', 'fuchsia', 'teal']
    
    for label in unique_labels:
        if label == -1:
            color = 'black'
            label_name = 'Noise'
        else:
            color = bright_colors[label % len(bright_colors)]
            label_name = f'Cluster {label}'
        
        plt.scatter(reduced[dbscan_labels == label, 0], reduced[dbscan_labels == label, 1], color=color, label=label_name, alpha=0.6)

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('DBSCAN Clustering Visualization')
    plt.legend()
    plt.grid()
    plt.show()

def HierarchicalClustering(data, kClusters):
    hierarchical = AgglomerativeClustering(n_clusters=kClusters, linkage='ward')
    labels = hierarchical.fit_predict(data)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)

    plt.figure(figsize=(12, 8))
    unique_labels = set(labels)

    bright_colors = [
        'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'pink', 'purple', 'lightblue',
        'lime', 'violet', 'turquoise', 'indigo', 'gold', 'salmon', 'coral', 'chartreuse', 'fuchsia', 'teal']
    
    for label in unique_labels:
        color = bright_colors[label % len(bright_colors)]
        label_name = f'Cluster {label}'
        
        cluster_points = reduced[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, label=label_name, alpha=0.6, edgecolors='k')

    plt.title(f'Hierarchical Clustering Visualization (Number of clusters: {kClusters})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.grid()
    plt.show()

def GMM(data, components):
    gmm = GaussianMixture(n_components=components, random_state=42)
    gmm_labels = gmm.fit_predict(data)

    # Reduce dimensions using PCA for visualization
    pca = PCA(n_components=2) 
    reduced_data = pca.fit_transform(data)

    # Plot the clusters
    plt.figure(figsize=(12, 8))
    unique_labels = set(gmm_labels)

    bright_colors = [
        'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'pink', 'purple', 'lightblue',
        'lime', 'violet', 'turquoise', 'indigo', 'gold', 'salmon', 'coral', 'chartreuse', 'fuchsia', 'teal']

    for label in unique_labels:
        color = bright_colors[label % len(bright_colors)]
        label_name = f'Cluster {label}'
        
        cluster_points = reduced_data[gmm_labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, label=label_name, alpha=0.6, edgecolors='k')

    plt.title(f'GMM Clustering Visualization (Number of clusters: {components})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.grid()
    plt.show()

if __name__ == '__main__':
    # Load preprocessed data
    yum_ingr = pd.read_pickle('../data/yummly_ingr.pkl')
    yum_ingrX = pd.read_pickle('../data/yummly_ingrX.pkl')
    yum_tfidf = pd.read_pickle('../data/yum_tfidf.pkl')

    # Ensure alignment of 'cuisine' column
    if 'cuisine' not in yum_ingrX.columns:
        yum_ingrX['cuisine'] = yum_ingr['cuisine']
    if 'recipeName' not in yum_ingrX.columns:
        yum_ingrX['recipeName'] = yum_ingr['recipeName']
    
    if 'cuisine' not in yum_tfidf.columns:
        yum_tfidf['cuisine'] = yum_ingr['cuisine']
    if 'recipeName' not in yum_tfidf.columns:
        yum_tfidf['recipeName'] = yum_ingr['recipeName']

    # convert data into numerical
    yum_ingrX_num = convertToNumerical(yum_ingrX)
    yum_tfidf_num = convertToNumerical(yum_tfidf)

    # scale the data
    scaler = StandardScaler()
    yum_ingrX_scaled = scaler.fit_transform(yum_ingrX_num)
    yum_tfidf_scaled = scaler.fit_transform(yum_tfidf_num)

    # DBSCAN
    DensityClustering(yum_ingrX_scaled, 9, 13)
    DensityClustering(yum_tfidf_scaled, 17, 15)

    # Hierarchical Clustering
    HierarchicalClustering(yum_ingrX_scaled, 8)
    HierarchicalClustering(yum_tfidf_scaled, 12)

    # GMM
    GMM(yum_ingrX_scaled, 12)
    GMM(yum_tfidf_scaled, 13)
