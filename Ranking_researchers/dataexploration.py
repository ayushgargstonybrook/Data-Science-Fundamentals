# -*- coding: utf-8 -*-
"""DataExploration.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yRrIz4qmsmuoY1Jwo55QiljEvjuuk4kh
"""

from google.colab import drive
drive.mount('/content/drive')

# Initial Python environment setup...
import numpy as np # linear algebra
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import os # reading the input files we have access to

print(os.listdir('/content/drive/My Drive/'))
pd.options.display.max_columns = 60

train_df =  pd.read_json('/content/drive/My Drive/dblp-ref-0.json',lines=True)
train_df.dtypes

"""There are 8 columns out of which 6 (abstract, authors, id, references, title, venue) are categorical. And remaining 2 (n_citiation, year) are numeric."""

train_df.info()

"""There are 1M entries in the dataset. The two columns (abstract, references) have some null values. Rest 6 columns don’t have any null values."""

train_df.describe()

train_df.head(5)

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

ax = sns.distplot(train_df['year'],kde=False,bins=50)
ax.set(xlabel='Publication Year', ylabel='Paper Count')
plt.title("Distrbution of papers published over years")
plt.show()

plottrain_df_morethan_50 = train_df[(train_df.n_citation>50)]
print (plottrain_df_morethan_50.shape)

print (len(train_df[(train_df.n_citation>50)]))
print (len(train_df[(train_df.n_citation>300)]))

print (sorted(train_df['n_citation'].unique()))

train_df_plotlessthan1 = train_df[(train_df.n_citation<1)]
print (len(train_df_plotlessthan1))

train_df_plotlessthan50 = train_df[(train_df.n_citation>0) & (train_df.n_citation<50)]
print (len(train_df_plotlessthan50))

train_df_plot50 = train_df[(train_df.n_citation==50)]
print (len(train_df_plot50))

train_df_plotmorethan_50 = train_df[(train_df.n_citation>50)]
print (len(train_df_plotmorethan_50))


train_df_ploteq1 = train_df[(train_df.n_citation==1)]
print (len(train_df_ploteq1))

xlabels = ["Zero", "1 to 49","50","More than 50"]
ycount = [len(train_df_plotlessthan1), len(train_df_plotlessthan50), len(train_df_plot50), len(train_df_plotmorethan_50)]

print (xlabels)
print (ycount)

train_df_plotmorethan_50 = train_df[(train_df.n_citation>10000)]
train_df_plotmorethan_50.head(3)

print (len(train_df['venue'].unique()))
train_df_plotmorethan_50 = train_df[(train_df.n_citation>70000)]
print (type(train_df_plotmorethan_50['venue']))
print (train_df_plotmorethan_50['venue'].empty)

(train_df['venue'].values == '').sum()

venuedf = train_df.groupby("venue")["id"].count()

print (type(venuedf))

venue_df = train_df.groupby('venue')[['id']].count() # Produces Pandas DataFrame

print(len(venue_df))
print(len(venue_df[(venue_df.id>5000)]))

df2 = venue_df[(venue_df.id>4000)].sort_values(by=['id'],ascending=False)
df2.rename(columns={"id": "Count of Papers"})