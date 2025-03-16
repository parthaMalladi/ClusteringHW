#INSTRUCTIONS:
##Order of running code files: recipe_cleanup.py-->graph_building.py-->recipe_clustering.py-->recipe_classification.py-->recipe_recommendation.py
#Performs t-SNE clustering for:Cuisines based on ingredients; Cuisines based on flavors and Generates interactive Bokeh plots
#Make sure to update the path of data frames
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

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

def performPCA(data, title):
    pca = PCA(n_components=0.95)
    reduced = pca.fit_transform(data)
    cumSum = np.cumsum(pca.explained_variance_ratio_)

    # find the components that explain 95% of the variance
    components = np.argmax(cumSum >= 0.95) + 1
    print("Number of components to explain 95 % variability:", components)

    # percentage variability for each component
    for i, variance in enumerate(pca.explained_variance_ratio_ * 100):
        if cumSum[i] <= 0.95:
            print(f"Component {i+1}: {variance:.2f}%")
        else:
            break
    
    print("")
    
    # plot components and variance
    plt.figure(figsize=(10, 8))
    plt.plot(cumSum, marker='.', linestyle='')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'PCA Graph for {title}')
    plt.legend()
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

    # apply PCA for yum_ingrX
    performPCA(yum_ingrX_scaled, "yum_ingrX")

    # apply PCA for yum_tfidf
    performPCA(yum_tfidf_scaled, "yum_tfidf")