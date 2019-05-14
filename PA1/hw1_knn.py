from __future__ import division, print_function

from typing import List

import numpy as np
import scipy

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: Complete the training function
    def train(self, features: List[List[float]], labels: List[int]):
        #raise NotImplementedError
        
        assert len(features) == len(labels)
        
        #store data only
        self.model_features = features
        self.model_labels = labels

        #print(self.model_features, model_lables)
     
    
    #TODO: Complete the prediction function
    def predict(self, features: List[List[float]]) -> List[int]:
        #raise NotImplementedError
        results:List[int] = []
        for single_point in features:
            single_predict = self.get_k_neighbors(single_point)
            if sum(single_predict)/len(single_predict) >= 0.5:
                results.append(1)
            else:
                results.append(0)
        
        return results
            
            
        
    #TODO: Complete the get k nearest neighbor function
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        #raise NotImplementedError
        distance_list = []
        
        for single_point,single_label in zip(self.model_features,self.model_labels):
            distance_list.append([self.distance_function(point,single_point),single_label])
        
        #print(distance_list)

        distance_list.sort()
        #print('distance_list:'+ str(distance_list[:self.k]))
        

        results:List[int] = []
        for _,label in distance_list[:self.k]:
            results.append(label)
        #print('knn points are: '+str(distance_list[:self.k]))
        return results


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
