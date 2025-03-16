'''Make recommendations based on flavor profile'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # `cPickle` is not needed in Python 3
from sklearn.metrics.pairwise import cosine_similarity

# Function to find and print similar dishes based on cosine similarity
def find_dishes(idx, similar_cuisine=False):
    if idx >= len(yum_ingr2):
        print(f"Error: Index {idx} is out of bounds. Maximum index is {len(yum_ingr2) - 1}.")
        return

    cuisine = yum_ingr2.iloc[idx]['cuisine']
    
    print(f"Dishes similar to {yum_ingr2.loc[idx, 'recipeName']} ({yum_ingr2.loc[idx, 'cuisine']})")

    # Find top 20 similar dishes (excluding the query dish itself)
    match = yum_ingr2.iloc[np.argsort(yum_cos[idx])[-21:-1][::-1]]

    # Filter based on cuisine preference
    if not similar_cuisine:
        submatch = match[match['cuisine'] != cuisine]
    else:
        submatch = match

    print()
    for i in submatch.index:
        print(f"{submatch.loc[i, 'recipeName']} ({submatch.loc[i, 'cuisine']}) (ID: {i})")

# Function to plot top 20 similar dishes
def plot_similar_dishes(idx, xlim):
    if idx >= len(yum_ingr2):
        print(f"Error: Index {idx} is out of bounds. Maximum index is {len(yum_ingr2) - 1}.")
        return

    match = yum_ingr2.iloc[np.argsort(yum_cos[idx])[-21:-1][::-1]]

    newidx = match.index.values
    match['cosine'] = yum_cos[idx][newidx]
    match['rank'] = range(1, len(newidx) + 1)

    label1 = match['cuisine'].tolist()
    label2 = match['recipeName'].tolist()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.stripplot(y='rank', x='cosine', data=match, jitter=0.05, hue='cuisine', size=15, orient="h", ax=ax)

    ax.set_title(f"{yum_ingr2.loc[idx, 'recipeName']} ({yum_ingr2.loc[idx, 'cuisine']})", fontsize=18)
    ax.set_xlabel('Flavor Cosine Similarity', fontsize=18)
    ax.set_ylabel('Rank', fontsize=18)
    ax.yaxis.grid(color='white')
    ax.xaxis.grid(color='white')

    for label, y, x in zip(label2, match['rank'], match['cosine']):
        ax.text(x + 0.001, y - 1, label, ha='left')

    ax.legend(loc='lower right', prop={'size': 14})
    ax.set_ylim([20, -1])
    ax.set_xlim(xlim)
    plt.show()

if __name__ == '__main__':
    # Load preprocessed data
    yum_ingr = pd.read_pickle('../data/yummly_ingr.pkl')
    yum_ingrX = pd.read_pickle('../data/yummly_ingrX.pkl')
    yum_tfidf = pd.read_pickle('../data/yum_tfidf.pkl')

    # Compute cosine similarity between flavor profiles
    yum_cos = cosine_similarity(yum_tfidf)

    # Reset index for consistency
    yum_ingr2 = yum_ingr.reset_index(drop=True)

    # Get dataset size
    num_samples = len(yum_ingr2)
    print(f"Dataset contains {num_samples} recipes.")

    # Check and correct indices
    if num_samples > 3900:
        idx = 3900
    else:
        idx = num_samples - 1  # Use the last available index if 3900 is out of range

    xlim = [0.91, 1.0]
    plot_similar_dishes(idx, xlim)

    if num_samples > 3315:
        idx = 3315
    else:
        idx = num_samples - 1  # Use the last available index if 3315 is out of range

    xlim = [0.88, 1.02]
    plot_similar_dishes(idx, xlim)
