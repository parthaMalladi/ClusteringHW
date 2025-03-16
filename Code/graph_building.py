'''Build graph with data from Yong Ahn's site
Do backbone extraction to simplify graph
Write csv file and do visualization in Cytoscape
'''
#INSTRUCTIONS:
#Order of running code files: recipe_cleanup.py-->graph_building.py-->recipe_clustering.py-->recipe_classification.py-->recipe_recommendation.py
#This file builds an ingredient-ingredient network graph; extracts the backbone structure and saves backbone.csv for Cytoscape visualization
import numpy as np
import pandas as pd
import networkx as nx

# Load ingredient and compound information
comp = pd.read_csv('../data/comp_info.tsv', index_col=0, sep='\t')
ingr_comp = pd.read_csv('../data/ingr_comp.tsv', sep='\t')
ingr = pd.read_csv('../data/ingr_info.tsv', index_col=0, sep='\t')

# Load ingredient-ingredient shared network
df = pd.read_csv('../data/srep00196-s2.csv', skiprows=4, header=None)
df.columns = ['ingr1', 'ingr2', 'shared']

# Merge with 'ingr' to get category information
df_category = pd.merge(df, ingr, left_on='ingr1', right_on='ingredient name').drop('ingredient name', axis=1)

# Load recipes dataset
recipe = pd.read_csv('../data/srep00196-s3.csv', skiprows=3, sep='\t')
recipe.columns = ['recipes']

# Extract ingredients from recipes
recipe['ingredients'] = recipe['recipes'].apply(lambda x: x.split(',')[1:])
all_ingredients = set()
recipe['ingredients'].explode().unique()  # More efficient way to extract unique ingredients

# Filter dataframe to only keep rows with ingredients present in the dataset
df_subset = df_category[df_category['ingr1'].isin(all_ingredients) & df_category['ingr2'].isin(all_ingredients)]

#### Build graph and extract backbone
G1 = nx.Graph()
weights = {}

for _, row in df_subset.iterrows():
    ingr1, ingr2, shared = row['ingr1'], row['ingr2'], row['shared']
    G1.add_edge(ingr1, ingr2)
    weights[(ingr1, ingr2)] = shared
    weights[(ingr2, ingr1)] = shared

# Extract backbone of graph using disparity filter
def extract_backbone(G, weights, alpha=0.04):
    keep_graph = nx.Graph()
    for n in G:
        k_n = len(G[n])
        if k_n > 1:
            sum_w = sum(weights[n, nj] for nj in G[n])
            for nj in G[n]:
                pij = weights[n, nj] / sum_w
                if (1 - pij) ** (k_n - 1) < alpha:  # edge is significant
                    keep_graph.add_edge(n, nj)
    return keep_graph

G1_backbone = extract_backbone(G1, weights)

# Convert weights into a list format
weights_subset = [(u, v, weights[u, v]) for (u, v) in G1_backbone.edges()]

df_weights = pd.DataFrame(weights_subset, columns=['ingr1', 'ingr2', 'weight'])
df_backbone = pd.merge(df_weights, ingr, left_on='ingr1', right_on='ingredient name').drop('ingredient name', axis=1)

# Efficiently create prevalence matrix
ingredients_list = list(all_ingredients)
recipe_matrix = pd.DataFrame(
    [{ing: ing in row for ing in ingredients_list} for row in recipe['ingredients']]
)

recipe_matrix.index = recipe.index  # Maintain index mapping

# Compute ingredient prevalence
prevalence = recipe_matrix.sum(axis=0) / recipe_matrix.sum().sum()
prevalence_df = prevalence.reset_index()
prevalence_df.columns = ['ingredient name', 'prevalence']

# Merge prevalence data with ingredient info
ingr_count = pd.merge(ingr, prevalence_df, on='ingredient name', how='right')

df_backbone = pd.merge(df_weights, ingr_count, left_on='ingr1', right_on='ingredient name').drop('ingredient name', axis=1)

# Save backbone data for visualization in Cytoscape
df_backbone.to_csv('../data/backbone.csv', index=False)

print(" Graph backbone extraction complete. Data saved in 'data/backbone.csv'.")
