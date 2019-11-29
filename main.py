import data
from lsh import LSH

if __name__ == '__main__':
	#get data from current given folder path
	getdata = data.Data()
	data = getdata.get_data()
	#get a list of questions
	listOfDocs = list(data['question'])
	#remove exact duplicates
	listOfDocs = list(set(listOfDocs))
	print(str(len(listOfDocs))+' questions extracted from dataset')
	#preprocess data to get a vector model ofquestions
	from preprocess import Preprocess
	preprocessObject = Preprocess(listOfDocs)
	questionTokens = preprocessObject.preprocess_data()
	print("Update: Preprocessing Complete")

	#perform  Similarity Search and get TF-IDF scores  of question tokens
	from similaritySearch import SimilaritySearch
	similaritySearchObj = SimilaritySearch(questionTokens)
	docList = similaritySearchObj.term_document_matrix

	print("Update: TF-IDF Generation Complete")

	print(docList.shape)
	#Now add all the docs to the lsh

	#reduce the size of the space matrix
	from scipy.sparse import csr_matrix
	matrix = csr_matrix(docList)

	print('Update: Converted  TF-IDF Matrix to Sparce matrix')

	lsh = LSH(8,
           matrix.shape[1],
           num_hashtables=10,
           storage_config={"dict":None})

	print("Update: LSH initialised")
	for ix in range(matrix.shape[0]):
		x = matrix.getrow(ix)
		lsh.index(x, extra_data=ix)

	print("Update: LSH indexing Complete")

	#get the buckets satisfying a given criteria
	lsh.getBestRepresentative(listOfDocs)


