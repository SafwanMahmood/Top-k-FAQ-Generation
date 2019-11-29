import pandas as pd 
import gzip
import glob

'''
  Retrives data from the current directory
'''

class Data:

	def __init__(self):
		return;

	def parse(self, filename):
		g = gzip.open(filename, 'rb') 
		for l in g: 
			yield eval(l)

	def get_data(self):
		df = {}  
		for filename in glob.glob('./'+'*.gz'):
			i = 0 
			for d in self.parse(filename): 
				df[i] = d 
				i += 1
				if(i == 1000):
					break
		return pd.DataFrame.from_dict(df, orient='index') 
