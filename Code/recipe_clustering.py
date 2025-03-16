#INSTRUCTIONS:
##Order of running code files: recipe_cleanup.py-->graph_building.py-->recipe_clustering.py-->recipe_classification.py-->recipe_recommendation.py
#Performs t-SNE clustering for:Cuisines based on ingredients; Cuisines based on flavors and Generates interactive Bokeh plots
#Make sure to update the path of data frames
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.palettes import Category20

# Take some regional cuisines, t-SNE clustering, and plotting
def tsne_cluster_cuisine(df, sublist):
    lenlist = [0]
    df_sub = df[df['cuisine'] == sublist[0]]
    lenlist.append(df_sub.shape[0])
    
    for cuisine in sublist[1:]:
        temp = df[df['cuisine'] == cuisine]
        df_sub = pd.concat([df_sub, temp], axis=0, ignore_index=True)
        lenlist.append(df_sub.shape[0])
    
    df_X = df_sub.drop(['cuisine', 'recipeName'], axis=1)
    print("Dataset shape for t-SNE:", df_X.shape)
    print("Cuisine sample counts:", lenlist)

    # Compute t-SNE embedding
    tsne = TSNE(metric='cosine', init='random', random_state=42).fit_transform(df_X)

    # Plot using seaborn color palette
    palette = sns.color_palette("hls", len(sublist))
    plt.figure(figsize=(10, 10))
    
    for i, cuisine in enumerate(sublist):
        plt.scatter(tsne[lenlist[i]:lenlist[i + 1], 0],
                    tsne[lenlist[i]:lenlist[i + 1], 1],
                    c=np.array(palette[i]).reshape(1, -1), label=cuisine)
    
    plt.legend()
    plt.title("t-SNE Clustering of Cuisines")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

# Interactive plot with Bokeh for visualization
def plot_bokeh(df, sublist, filename):
    lenlist = [0]
    df_sub = df[df['cuisine'] == sublist[0]]
    lenlist.append(df_sub.shape[0])
    
    for cuisine in sublist[1:]:
        temp = df[df['cuisine'] == cuisine]
        df_sub = pd.concat([df_sub, temp], axis=0, ignore_index=True)
        lenlist.append(df_sub.shape[0])
    
    df_X = df_sub.drop(['cuisine', 'recipeName'], axis=1)
    print("Dataset shape for Bokeh:", df_X.shape)
    print("Cuisine sample counts:", lenlist)

    tsne = TSNE(metric='cosine', init='random', random_state=42).fit_transform(df_X)

    # palette = ['red', 'green', 'blue', 'yellow']
    palette = sns.color_palette("hls", len(sublist)).as_hex()

    colors = []
    for i in range(len(sublist)):
        colors.extend([palette[i]] * (lenlist[i + 1] - lenlist[i]))

    df_sub['colors'] = colors

    # Prepare Bokeh plot
    output_file(filename)
    source = ColumnDataSource(
        data=dict(
            x=tsne[:, 0], 
            y=tsne[:, 1],
            cuisine=df_sub['cuisine'],
            recipe=df_sub['recipeName'],
            color=df_sub['colors']  
        )
    )

    hover = HoverTool(tooltips=[("cuisine", "@cuisine"), ("recipe", "@recipe")])
    
    p = figure(width=1000, height=1000, tools=[hover], title="Flavor Clustering")
    p.circle('x', 'y', size=10, source=source, fill_color='color')  

    show(p)

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

    # Select cuisines for visualization
    sublist = ['Italian', 'French', 'Japanese', 'Indian']

    # Select all cuisines
    allCusines = yum_ingrX["cuisine"].unique().tolist()

    # t-SNE Clustering with Ingredients
    tsne_cluster_cuisine(yum_ingrX, allCusines)

    # t-SNE Clustering with Flavors
    tsne_cluster_cuisine(yum_tfidf, allCusines)

    # Interactive Bokeh Visualization
    plot_bokeh(yum_tfidf, allCusines, 'flavor_clustering.html')
    plot_bokeh(yum_ingrX, allCusines, 'ingredient_clustering.html')
