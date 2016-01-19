import pandas as pd 
import os 
import pyprind

pbar = pyprind.ProgBar(50000)

labels = {'pos': 1, 'neg': 0}

df = pd.DataFrame()

for s in {'test', 'train'}:
	for l in ('pos', 'neg'):
		path = './aclImdb/%s/%s' % (s, l)
		for f in os.listdir(path):
			with open(os.path.join(path, f), 'r') as infile:
				txt = infile.read()
			df = df.append([[txt, labels[l]]], ignore_index = True)
			pbar.update()
df.columns = ['review', 'sentiment']