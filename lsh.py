from __future__ import print_function

import os
import json
import numpy as np
from scipy import sparse

from storage import storage, serialize, deserialize

class LSH(object):
    """ LSH implments locality sensitive hashing using random projection for
    input vectors of dimension `input_dim`.

    Attributes:

    :param hash_size:
        The length of the resulting binary hash in integer. E.g., 32 means the
        resulting binary hash will be 32-bit long.
    :param input_dim:
        The dimension of the input vector. This can be found in your sparse
        matrix by checking the .shape attribute of your matrix. I.E.,
            `csr_dataset.shape[1]`
    :param num_hashtables:
        (optional) The number of hash tables used for multiple look-ups.
        Increasing the number of hashtables increases the probability of
        a hash collision of similar documents, but it also increases the
        amount of work needed to add points.
    :param storage_config:
        (optional) A dictionary of the form `{backend_name: config}` where
        `backend_name` is the either `dict`, `berkeleydb`, `leveldb` or
        `redis`. `config` is the configuration used by the backend.
        Example configs for each type are as follows:
        `In-Memory Python Dictionary`:
            {"dict": None} # Takes no options
        `Redis`:
            `{"redis": {"host": hostname, "port": port_num}}`
            Where `hostname` is normally `localhost` and `port` is normally 6379.
        `LevelDB`:
            {'leveldb':{'db': 'ldb'}}
            Where 'db' specifies the directory to store the LevelDB database.
        `Berkeley DB`:
            {'berkeleydb':{'filename': './db'}}
            Where 'filename' is the location of the database file.
        NOTE: Both Redis and Dict are in-memory. Keep this in mind when
        selecting a storage backend.
    :param matrices_filename:
        (optional) Specify the path to the compressed numpy file ending with
        extension `.npz`, where the uniform random planes are stored, or to be
        stored if the file does not exist yet.
    :param overwrite:
        (optional) Whether to overwrite the matrices file if it already exist.
        This needs to be True if the input dimensions or number of hashtables
        change.
    """

    def __init__(self, hash_size, input_dim, num_hashtables=1,
                 storage_config=None, matrices_filename=None, overwrite=False):

        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables

        if storage_config is None:
            storage_config = {'dict': None}
        self.storage_config = storage_config

        if matrices_filename and not matrices_filename.endswith('.npz'):
            raise ValueError("The specified file name must end with .npz")
        self.matrices_filename = matrices_filename
        self.overwrite = overwrite

        self._init_uniform_planes()
        self._init_hashtables()

    def _init_uniform_planes(self):
        """ Initialize uniform planes used to calculate the hashes

        if file `self.matrices_filename` exist and `self.overwrite` is
        selected, save the uniform planes to the specified file.

        if file `self.matrices_filename` exist and `self.overwrite` is not
        selected, load the matrix with `np.load`.

        if file `self.matrices_filename` does not exist and regardless of
        `self.overwrite`, only set `self.uniform_planes`.
        """

        if "uniform_planes" in self.__dict__:
            return

        if self.matrices_filename:
            file_exist = os.path.isfile(self.matrices_filename)
            if file_exist and not self.overwrite:
                try:
                    # TODO: load sparse file
                    npzfiles = np.load(self.matrices_filename)
                except IOError:
                    print("Cannot load specified file as a numpy array")
                    raise
                else:
                    npzfiles = sorted(list(npzfiles.items()), key=lambda x: x[0])
                    # TODO: to sparse
                    self.uniform_planes = [t[1] for t in npzfiles]
            else:
                self.uniform_planes = [self._generate_uniform_planes()
                                       for _ in range(self.num_hashtables)]
                try:
                    np.savez_compressed(self.matrices_filename,
                                        *self.uniform_planes)
                except IOError:
                    print("IOError when saving matrices to specificed path")
                    raise
        else:
            self.uniform_planes = [self._generate_uniform_planes()
                                   for _ in range(self.num_hashtables)]

    def _init_hashtables(self):
        """ Initialize the hash tables such that each record will be in the
        form of "[storage1, storage2, ...]" """

        self.hash_tables = [storage(self.storage_config, i)
                            for i in range(self.num_hashtables)]

    def _generate_uniform_planes(self):
        """ Generate uniformly distributed hyperplanes and return it as a 2D
        numpy array.
        """
        dense_planes = np.random.randn(self.hash_size, self.input_dim)
        return sparse.csr_matrix(dense_planes)

    def _hash(self, planes, input_point):
        """ Generates the binary hash for `input_point` and returns it.

        :param planes:
            The planes are random uniform planes with a dimension of
            `hash_size` * `input_dim`.
        :param input_point:
            A scipy sparse matrix that contains only numbers.
            The dimension needs to be 1 * `input_dim`.
        """
        try:
            input_point = input_point.transpose()
            projections = planes.dot(input_point)

        except TypeError as e:
            print("""The input point needs to be an array-like object with
                  numbers only elements""")
            raise
        except ValueError as e:
            print(("""The input point needs to be of the same dimension as
                  `input_dim` when initializing this LSH instance""", e))
            raise
        else:
            return "".join(['1' if i > 0 else '0' for i in projections])

    def _as_np_array(self, serial_or_sparse):
        """ Takes either a serialized data structure, a sparse matrix, or tuple
        that has the original input points stored, and returns the original
        input point (a 1 x N sparse matrix).
        """
        # if we get a plain sparse matrix, return it (it's the point itself)
        if sparse.issparse(serial_or_sparse):
            return serial_or_sparse

        # here we have a serialized pickle object
        if isinstance(serial_or_sparse, str):
            try:
                deserial = deserialize(serial_or_sparse)
            except TypeError:
                print("The value stored is not deserializable")
                raise
        else:
            # If extra_data exists, `tuples` is the entire
            # (point:sparse, extra_daa). Otherwise (i.e., extra_data=None),
            # return the point stored as a tuple
            deserial = serial_or_sparse

        # if we deserialized it, we might have the sparse now
        if sparse.issparse(deserial):
            return deserial

        if isinstance(deserial[0], tuple):
            # extra data was supplied, return point
            return tuples[0]

        elif isinstance(deserial, (tuple, list)):
            try:
                return deserial[0]
            except ValueError as e:
                print(("The input needs to be an array-like object", e))
                raise
        else:
            raise TypeError("the input data is not supported")

    def index(self, input_point, extra_data=None):
        """ Index a single input point by adding it to the selected storage.

        If `extra_data` is provided, it will become the value of the dictionary
        {input_point: extra_data}, which in turn will become the value of the
        hash table.

        :param input_point:
            A sparse CSR matrix. The dimension needs to be 1 * `input_dim`.
        :param extra_data:
            (optional) A value to associate with the point. Commonly this is
            a target/class-value of some type.
        """

        assert sparse.issparse(input_point), "input_point needs to be sparse"

        # NOTE: there was a bug with 0-equal extra_data
        # we need to allow blank extra_data if it's provided
        if not isinstance(extra_data, type(None)):
            # NOTE: needs to be tuple so it's set-hashable
            value = (input_point, extra_data)
        else:
            value = input_point

        for i, table in enumerate(self.hash_tables):
            table.append_val(
                self._hash(self.uniform_planes[i], input_point),
                value)

    def _string_bits_to_array(self, hash_key):
        """ Take our hash keys (strings of 0 and 1) and turn it
        into a numpy matrix we can do calculations with.

        :param hash_key
        """
        return np.array( [ float(i) for i in hash_key])

    def cosine_similarity_array(self, document_vector_1, document_vector_2):
        document_vector_1 = np.array(document_vector_1)

    def distance(self,document_id_1,document_id_2):
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(document_id_1,document_id_2)

    def get_question_with_maximum_average_cosine_similarity(self,points, d_func):
        dic_score_list = {}
        dic_score_list_index = {}
        for (array,index) in points:
            dic_score_list[index] = []
            dic_score_list_index[index] = []
            for (array2,index2) in points:
                dist  = self.distance(array.toarray(),array2.toarray())
                dic_score_list[index].append(dist)
                dic_score_list_index[index].append(index2)
        
        average_dic_score = -1.0
        average_dic_score_index = -1

        for index in dic_score_list.keys():
            mean = np.mean(np.array(dic_score_list[index]))
            if average_dic_score<mean:
                average_dic_score = mean
                average_dic_score_index = index
        return (average_dic_score,average_dic_score_index)

    '''
        gets best replsentatives of the bucket TODO: need to fix the threshold in terms of 
        percentage
    '''
    def getBestRepresentative(self, originalData, bucket_capacity_threshold = 10, cosine_threshold = 0.5):
        candidates = []
        d_func = LSH.cosine_dist

        tableSet = set()
        greaterThanThreshold = []
        maxLength_id = (0,-1)
        for i, table in enumerate(self.hash_tables):
            table_keys = table.keys()
            for key in table_keys:
                if len(table.get_list(key)) > bucket_capacity_threshold:
                    greaterThanThreshold.append({(key,table):table.get_list(key)})
                    tableSet.add(table)
        meanArray = []
        for dic in greaterThanThreshold:
            (key,table) = dic.keys()[0]
            (mean,index_of_best) = self.get_question_with_maximum_average_cosine_similarity(table.get_list(key), d_func)
            if mean>.4:
                print('\n\n------------',key,'------------')
                print('\n\n------------best------------')
                print(originalData[index_of_best],index_of_best,mean)
                print('------------best------------\n\n')
                for (array, index) in table.get_list(key):
                    print(originalData[index],  index)
                print('------------',key,'------------\n\n')
            # print(mean,originalData[index_of_best])
            # meanArray.append(mean)
        # 

    @staticmethod
    def hamming_dist(sparse1, sparse2):
        return (sparse1 != sparse2).sum()

    @staticmethod
    def euclidean_dist(x, y):
        diff = x - y
        return sparse.csr_matrix.sqrt( diff.dot(diff))

    @staticmethod
    def euclidean_dist_square(x, y):
        diff = x - y
        if diff.nnz == 0:
            return 0.0
        result = diff.dot(diff.transpose())
        return result.data[0]

    @staticmethod
    def euclidean_dist_centred(x, y):
        diff = x.mean() - y.mean()
        return diff.dot( diff)

    @staticmethod
    def l1norm_dist(x, y):
        return abs(x - y).sum()

    @staticmethod
    def cosine_dist(x, y):
        return 1 - x.dot(y) / ((x.dot(x) * y.dot(y)) ** 0.5)
