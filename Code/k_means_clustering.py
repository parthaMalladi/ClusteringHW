#INSTRUCTIONS:
##Order of running code files: recipe_cleanup.py-->graph_building.py-->recipe_clustering.py-->recipe_classification.py-->recipe_recommendation.py
#Performs t-SNE clustering for:Cuisines based on ingredients; Cuisines based on flavors and Generates interactive Bokeh plots
#Make sure to update the path of data frames
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import seaborn as sns

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

def kMeansClustering(data, title, kClusters):
    fig = plt.subplots(nrows = 1, ncols = 2, figsize = (20,5))

    # ELBOW METHOD
    plt.subplot(1,2,1)

    # map to store the inertia
    inertia = {}

    # calculate inertia for each cluster
    for k in range(1, kClusters + 1):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
        inertia[k] = kmeans.inertia_
    
    # plot elbow graph
    clusters = list(inertia.keys())
    sse = list(inertia.values())
    sns.lineplot(x = clusters, y = sse, marker="o")
    plt.title(f'Elbow Method for {title}')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.xticks(range(1, kClusters))

    # SILHOUETTE SCORE METHOD
    plt.subplot(1,2,2)

    # list to store silhouette scores
    silScore = []

    # calculate silhouette scores
    for k in range(2, kClusters + 1):
        kmeans = KMeans(n_clusters = k).fit(data)
        labels = kmeans.labels_
        silScore.append(silhouette_score(data, labels, metric = 'euclidean'))
    
    # plot silhouette graph
    sns.lineplot(x = range(2,kClusters + 1), y = silScore, marker="o")
    plt.title(f'Silhouette Score Method for {title}')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.xticks(range(2, kClusters + 1))
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
    
    # See number of rows and columns
    print("\nyum_ingrX file:", yum_ingrX.shape)
    print("\nyum_tfidf file:", yum_tfidf.shape)

    # convert data into numerical
    yum_ingrX_num = convertToNumerical(yum_ingrX)
    yum_tfidf_num = convertToNumerical(yum_tfidf)

    # scale the data
    scaler = StandardScaler()
    yum_ingrX_scaled = scaler.fit_transform(yum_ingrX_num)
    yum_tfidf_scaled = scaler.fit_transform(yum_tfidf_num)

    # elbow and silhouette graphs
    kMeansClustering(yum_ingrX_scaled, "yum_ingrX", 20)
    kMeansClustering(yum_tfidf_scaled, "yum_tfidf", 20)