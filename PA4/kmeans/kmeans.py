import numpy as np
import time as tm
import copy

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.

    # raise Exception(
    #          'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')

    # print(generator())
    
    _, D = x.shape
    rand_num = generator.randint(n)
    last_cent = x[rand_num,:]
    max_distance = 0
    max_ind = rand_num
    time = 0
    res_centers_ind = [rand_num]
    res_centers_data = np.zeros([1,D])
    res_centers_data[0,:] = x[rand_num,:]
    # print(res_centers_data)

    while time < n_cluster - 1:
        max_distance = []
        # find largest distance
        for i,row in enumerate(x):

            if i in res_centers_ind:
                continue
            
            cur_distance = min(np.sum(row**2 + res_centers_data**2 - 2*row*res_centers_data,axis=1))
            max_distance.append(cur_distance)

        max_ind = np.argmax(max_distance)

        res_centers_data = np.append(res_centers_data,np.reshape(x[max_ind,:],(1,D)),axis=0)
        res_centers_ind.append(max_ind)

        time +=1

    
    centers = res_centers_ind
    

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #      'Implement fit function in KMeans class')

        #extract from x for centers
        centroids = np.zeros([self.n_cluster,D])
        for n,index in enumerate(self.centers):
            centroids[n] = np.add(centroids[n], x[index,:])

        init_sum = 0
        assign_sum = 0
        recenter_sum = 0
        calculating_sum = 0

        time = 0
        last_obj = float('inf')
        while time < self.max_iter:

            mystart = tm.time()
            # fixed centers and assign points to closest center
            y = []
            # clusters = []
            
            init = tm.time()
            init_sum += init - mystart
            #print("time takes for init phase:" + str(init-mystart))
            
            b2 = centroids ** 2
            for row in x:
                close_cent_ind = 0
                close_cent_ind = np.argmin(np.sum(row**2 + b2 - 2*row*centroids,axis=1))
                y.append(close_cent_ind)
            
            assign = tm.time()
            assign_sum += assign - init
            
            # membership has created y is index array
            # fixed membership and update centers
            
            last_centroids = copy.deepcopy(centroids)

            # if 0 not in points_sum:
            for i in range(self.n_cluster):
                points = [x[j] for j in range(len(x)) if y[j] == i]
                if len(points) == 0:
                    continue
                centroids[i] = np.mean(points,axis=0)
            recenter = tm.time()
            recenter_sum += recenter - assign

            # obj = 0
            obj = np.sum(centroids**2+last_centroids**2-2*centroids*last_centroids)

            calculating = tm.time()
            calculating_sum += calculating - recenter
            #print("time takes for calculting phase:"+str(calculating - recenter))
            
            print("current obj is:" + str(obj))
            if obj <= 0.00001 or last_obj - obj <= 0:
                break 

            last_obj = obj

            time +=1

        y = []
        b2 = centroids ** 2
        for row in x:
            close_cent_ind = 0
            close_cent_ind = np.argmin(np.sum(row**2 + b2 - 2*row*centroids,axis=1))
            y.append(close_cent_ind)
        #y = np.array(last_y)
        y = np.array(y)

        np.set_printoptions(threshold=np.inf)

        print("finish at tieration:" + str(time))
        print("time takes for init phase:" + str(init_sum))
        print("time takes for assign phase:"+str(assign_sum))
        print("time takes for recenter phase:"+str(recenter_sum))
        print("time takes for calculting phase:"+str(calculating_sum))
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
                ????
                self.centroid_labels: (n_cluster,) array
                ????
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #      'Implement fit function in KMeansClassifier class')

        model = KMeans(self.n_cluster,self.max_iter,self.e,self.generator)
        centroids, membership, _ = model.fit(x,centroid_func)

        # array with size(n_cluster)
        votes = [dict() for x in range(self.n_cluster)]
        for index, vote in zip(membership,y):
            votes[index][vote] = votes[index].get(vote,0) + 1

        centroid_labels = []
        for vote in votes:
            ranking_arr = self.rank_majority_dic(vote)

            if len(ranking_arr) == 0:
                centroid_labels.append(0)
                continue
            
            i = 0
            while i < len(ranking_arr) and ranking_arr[i] in centroid_labels:
                i+=1
            
            if i != len(ranking_arr):
                centroid_labels.append(ranking_arr[i])
            else:
                centroid_labels.append(0)

        # print(centroid_labels)

        centroid_labels = np.array(centroid_labels)
        centroids = np.array(centroids)

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)
    
    def rank_majority_dic(self, dic):        
        sorted_dict = dict(sorted(dic.items(), key = lambda t: t[0]))
        ranking_dict = sorted(sorted_dict.items(), key = lambda t: t[1],reverse=True)

        ranking_array = []
        for k,_ in ranking_dict:
            ranking_array.append(k)

        return ranking_array
            
    

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #      'Implement predict function in KMeansClassifier class')
        labels = []
        for row in x:
            min_distance = float('inf')
            min_index = 0
            for i,center in enumerate(self.centroids):
                
                if min_distance > np.sum(row **2 + center **2 - 2 * row * center):
                    min_distance = np.sum(row **2 + center **2 - 2 * row * center)
                    min_index = i
            labels.append(self.centroid_labels[min_index])
      
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    # raise Exception(
    #          'Implement transform_image function')

    M,N,_ = np.array(image).shape
    new_im = np.zeros((M,N,3))

    for i in range(M):
        for j in range(N):
            pixel = image[i][j]
            cluster_index = np.argmin(np.sum(pixel**2 + code_vectors**2 - 2*pixel*code_vectors,axis=1))
            new_im[i][j] = code_vectors[cluster_index]

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

