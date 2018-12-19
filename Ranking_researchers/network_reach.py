# -*- coding: utf-8 -*-
"""Network-Reach.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bf3WaVJIwlLqjYjg4e4Lha3hyIkEqCVX
"""

#Import
import pandas as pd
import numpy as np

import json
from pandas.io.json import json_normalize
import networkx as nx

from google.colab import drive
drive.mount('/content/drive')

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

df = pd.read_json('/content/drive/My Drive/dblp-ref-0.json',lines=True)

df[1]

df.head(1)

pip install git+https://github.com/OrganicIrradiation/scholarly.git

import scholarly

print(next(scholarly.search_author('Steven A. Cholewiak')))

df_copy = df.copy()

def calculate_page_rank_utility(df_pageRank):
    from  itertools import chain
    df_pageRank_flatten = pd.DataFrame({
            "abstract": np.repeat(df_pageRank.abstract.values, df_pageRank.references.str.len()),
            "authors": np.repeat(df_pageRank.authors.values, df_pageRank.references.str.len()),
            "id": np.repeat(df_pageRank.id.values, df_pageRank.references.str.len()),
            "n_citation": np.repeat(df_pageRank.n_citation.values, df_pageRank.references.str.len()),
            "references": list(chain.from_iterable(df_pageRank.references)),
            "title": np.repeat(df_pageRank.title.values, df_pageRank.references.str.len()),
            "venue": np.repeat(df_pageRank.venue.values, df_pageRank.references.str.len()),
            "year": np.repeat(df_pageRank.year.values, df_pageRank.references.str.len()),        
            "pageRankValue": np.repeat(df_pageRank.pageRankValue.values, df_pageRank.references.str.len()),
            "totalOutbounds": np.repeat(df_pageRank.totalOutbounds.values, df_pageRank.references.str.len()),
            "ratio": np.repeat(df_pageRank.ratio.values, df_pageRank.references.str.len())        
    })

    df_pageRank_flatten_groupby = df_pageRank_flatten.groupby('references').agg({'ratio' : 'sum'})
    df_pageRank_flatten_groupby['id'] = df_pageRank_flatten_groupby.index
    df_pageRank_flatten_groupby['newPageRankValue'] = df_pageRank_flatten_groupby['ratio']

    df_pageRank_new = pd.merge(df_pageRank, df_pageRank_flatten_groupby[['id', 'newPageRankValue']], how='left', on='id')
    df_pageRank_new.pageRankValue = df_pageRank_new.newPageRankValue
    df_pageRank_new['ratio'] = df_pageRank_new.pageRankValue / df_pageRank_new.totalOutbounds
    df_pageRank_new = df_pageRank_new.drop(columns = ['newPageRankValue'])

    return df_pageRank_new

def calculate_page_rank (df, iteration):
    for i in range (0, iteration):
        df = calculate_page_rank_utility(df)
    return df

df_page_rank_actual = df.copy()

df_page_rank_actual.shape

#placing [self] where references is nan
df_page_rank_actual.loc[df_page_rank_actual['references'].isnull(),['references']] = df_page_rank_actual.loc[df_page_rank_actual['references'].isnull(),'references'].apply(lambda references : ['self'])

#also do for empty array
df_page_rank_actual.references = df_page_rank_actual.references.apply(lambda y: ['self'] if len(y)==0 else y)

#without [self], empty array, on flatten entries with empty array will be deleted

from  itertools import chain
df_page_rank_actual_flatten_references = pd.DataFrame({
            "id": np.repeat(df_page_rank_actual.id.values, df_page_rank_actual.references.str.len()),
            "references": list(chain.from_iterable(df_page_rank_actual.references))
    })

df_page_rank_actual_flatten_references.shape

df_page_rank_actual_flatten_references.references.unique().shape

unique_paper_id = df_page_rank_actual_flatten_references.id.unique()

unique_reference_id = df_page_rank_actual_flatten_references.references.unique()

print (unique_paper_id.shape)

print (unique_reference_id.shape)

diff_set = set(unique_reference_id)-set(unique_paper_id)

len(diff_set)

df_page_rank_actual_flatten_references = df_page_rank_actual_flatten_references[~df_page_rank_actual_flatten_references['references'].isin(diff_set)]

df_page_rank_actual_flatten_references.shape

print (df_page_rank_actual_flatten_references.references.unique().shape)

print (df_page_rank_actual_flatten_references.shape)

df_page_rank_actual_flatten_references.head(5)

df_page_rank_actual_flatten_references[df_page_rank_actual_flatten_references.id=='6f52f995-7c4c-4a92-83aa-d1c9fbd2465c']

df_page_rank_actual_flatten_references.to_pickle("/content/drive/My Drive/df_page_rank_actual_flatten_references.pkl")

df_page_rank_actual_flatten_references = pd.read_pickle("/content/drive/My Drive/df_page_rank_actual_flatten_references.pkl")
df_page_rank_actual_flatten_references['reach'] = 0

df_page_rank_actual_flatten_references[df_page_rank_actual_flatten_references.id=='6f52f995-7c4c-4a92-83aa-d1c9fbd2465c']

df_page_rank_actual_flatten_references.shape

df_page_rank_actual_flatten_references_small = df_page_rank_actual_flatten_references[0:1000]

df_page_rank_actual_flatten_references_small.shape

df_page_rank_actual_flatten_references_small.head(5)

author_graph = nx.Graph()

def update_author_graph(coauthors):  
  edges=[]
  for author1 in coauthors:
    for author2 in coauthors:
      if author1 != author2:        
        edge = (author1,author2)
        edges.append(edge)
  author_graph.add_edges_from(edges)
  return coauthors

def max_collaboration_distance(author_set_1,author_set_2):
  farthestAuthorDist = 0
  for author1 in author_set_1:
    for author2 in author_set_2:
      try:
        #print ("type of author1:",type(author1))
        dist = nx.shortest_path_length(author_graph, source=author1, target=author2)
      except nx.NetworkXNoPath:
        dist = float("inf")
        #print ("No path exists between "+author1+" and "+author2)
        pass
      except nx.NodeNotFound:
        dist = 0
        #print ("Node not found "+author1+" or "+author2)
        pass
      #print ("Distance between "+author1 +" and "+ author2 +" is : "+ str(dist))
      farthestAuthorDist = max(farthestAuthorDist, dist)
  return farthestAuthorDist

df.shape

train_df_authors_small = df[0:1000000]
[x['authors'].apply(update_author_graph) for x in np.split(train_df_authors_small, np.arange(5, len(train_df_authors_small), 5))]

nx.write_gpickle(author_graph, "/content/drive/My Drive/author_graph.gpickle")

author_graph = nx.read_gpickle("/content/drive/My Drive/author_graph.gpickle")

def update_max_score(row):  
  
  authors1_series =  df[df.id==row['references']]['authors']
  
  #print ("\n Paper id : " + row['references'])
  author_set_1 = get_list_of_authors(authors1_series)
  #print (" Authors : ",author_set_1)
  
  
  authors2_series = df[df.id==row['id']]['authors']
  #print (" Citation id : ",row['id'])
  author_set_2 = get_list_of_authors(authors2_series)
  #print (" Authors : ", author_set_2)
  
  #print ("Reach :",max_collaboration_distance(author_set_1,author_set_2))
  
  row['reach'] = max_collaboration_distance(author_set_1,author_set_2)
    
  return row
  
  
#This function takes a object of type Series and converts it to a list of authors
def get_list_of_authors(author_series):
  authors = []
  for key,value in author_series.iteritems():
    for author in value:
      authors.append(author)    
  return authors

df_page_rank_actual_flatten_references.shape

df_page_rank_actual_flatten_references.head(5)

low=112700
high=112800
for i in range(200):
  print (str(low)+", "+str(high))
  #df_page_rank_actual_flatten_references_small = df_page_rank_actual_flatten_references[low:high]
  df_page_rank_actual_flatten_references_small = df_page_rank_actual_flatten_references[low:high]
  reach_df = df_page_rank_actual_flatten_references_small.apply(update_max_score, axis=1)
  reach_df_compressed = reach_df[['id','reach']]
  max_reach_df_compressed = reach_df_compressed.groupby('id',as_index=False)['reach'].max()
  filename = "/content/drive/My Drive/author_graph/max_reach_df_compressed_" + str(low)+"_"+str(high)+".pkl"
  print ("will save df to:", filename)
  max_reach_df_compressed.to_pickle(filename)
  low+=100
  high+=100

import os
os.listdir("/content/drive/My Drive/author_graph/")

dirs = os.listdir("/content/drive/My Drive/author_graph/")

df_list=[]
for file in dirs:
  #print (file)
  #print (type(file))
  reach_df = pd.read_pickle("/content/drive/My Drive/author_graph/"+file)
  #print (reach_df.shape)
  df_list.append(reach_df)
#print (df_list)

bigdf = pd.concat(df_list)

bigdf.shape

bigdf.head(5)

df_page_rank_actual_flatten_references_small.shape

df_page_rank_actual_flatten_references_small = df_page_rank_actual_flatten_references[1000:2000]

reach_df = df_page_rank_actual_flatten_references_small.apply(update_max_score, axis=1)

df_page_rank_actual_flatten_references_small.head(5)

reach_df.shape

reach_df.head(5)

reach_df_compressed = reach_df[['id','reach']]

reach_df_compressed.head(5)

max_reach_df_compressed = reach_df_compressed.groupby('id',as_index=False)['reach'].max()

max_reach_df_compressed.shape

max_reach_df_compressed.head(5)

max_reach_df_compressed.info()

max_reach_df_compressed.to_pickle("/content/drive/My Drive/max_reach_df_compressed_2.pkl")

max_reach_df_compressed_1 = pd.read_pickle("/content/drive/My Drive/max_reach_df_compressed_1.pkl")
max_reach_df_compressed_2 = pd.read_pickle("/content/drive/My Drive/max_reach_df_compressed_2.pkl")

bigdf = pd.concat([max_reach_df_compressed_1,max_reach_df_compressed_2])

bigdf.shape

bigdf.to_pickle("/content/drive/My Drive/bigdf_1.pkl")

print (len(bigdf['id'].unique()))

bigdf.groupby('reach').count().sort_values(by=['id'],ascending=False)

bigdf.head(5)

bigdf_grouped_by_id = bigdf.groupby('id',as_index=False)['reach'].max()

bigdf_grouped_by_id.shape

bigdf_grouped_by_id.head(5)

max_reach_df_compressed_2.head(5)

reach_df_grouped_by_id_series = reach_df_compressed.groupby(['id'])['reach'].max()

reach_df_grouped_by_id_series.shape

reach_df_grouped_by_id_series

print (type(reach_df_grouped_by_id_series))

reach_df_grouped_by_id = reach_df_grouped_by_id_series.to_frame()

print (type(reach_df_grouped_by_id))

reach_df_grouped_by_id.shape

reach_df_grouped_by_id.head(5)

for i, v in reach_df_grouped_by_id_series.iteritems():
    print('index: ', i, 'value: ', v)
    #print (type(i))
    #print (type(v))
    reach_df_grouped_by_id[]

nx.draw(author_graph,node_size=100,with_labels=False)

author_series = df[df.id=='3a926fef-7422-4654-8776-8e31b45be563']['authors']
print (get_list_of_authors(author_series))
list = ['Veronica Sundstedt', 'Alan Chalmers', 'Philippe Martinez']
print (list)

df_page_rank_actual_flatten_references = df_page_rank_actual_flatten_references.groupby('id').agg({'references':lambda x: list(x)})

df_page_rank_actual_flatten_references.shape

df_page_rank_actual_flatten_references['id'] = df_page_rank_actual_flatten_references.index

df_page_rank_actual_flatten_references['references2'] = df_page_rank_actual_flatten_references['references']

df_page_rank_actual_flatten_references.shape

df_page_rank_actual = pd.merge(df_page_rank_actual, df_page_rank_actual_flatten_references[['id', 'references2']], how='left', on='id')

df_page_rank_actual.shape

df_page_rank_actual['references'] = df_page_rank_actual['references2']

df_page_rank_actual = df_page_rank_actual.drop(columns = ['references2'])

df_page_rank_actual.shape

#add 3 columns required for calculating page rank

df_page_rank_actual['pageRankValue'] = 100
df_page_rank_actual['totalOutbounds'] = df_page_rank_actual.references.str.len()
df_page_rank_actual['ratio'] = df_page_rank_actual.pageRankValue / df_page_rank_actual.totalOutbounds

df_page_rank_actual.shape

df_page_rank_actual['pageRankValue'].sum()

df_page_rank_actual_iterate1 = calculate_page_rank(df_page_rank_actual, 1)
#error because of NaN in references data .... after left join .... some rows were not found

df_page_rank_actual.head()

#placing [self] where references is nan
df_page_rank_actual.loc[df_page_rank_actual['references'].isnull(),['references']] = df_page_rank_actual.loc[df_page_rank_actual['references'].isnull(),'references'].apply(lambda references : ['self'])

#also do for empty array
df_page_rank_actual.references = df_page_rank_actual.references.apply(lambda y: ['self'] if len(y)==0 else y)

#without [self], empty array, on flatten entries with empty array will be deleted

df_page_rank_actual.head()

df_page_rank_actual_iterate1 = calculate_page_rank(df_page_rank_actual, 1)

df_page_rank_actual_iterate1.shape

df_page_rank_actual_iterate1['pageRankValue'].sum()

df_page_rank_actual_iterate1.head()

df_page_rank_actual_iterate2 = calculate_page_rank(df_page_rank_actual_iterate1, 1)

df_page_rank_actual_iterate2['pageRankValue'].sum()

df_page_rank_actual = df_copy.copy()

df_page_rank_actual['references'].isnull().count()

df_page_rank_actual['references'].count()

df_page_rank_actual.shape