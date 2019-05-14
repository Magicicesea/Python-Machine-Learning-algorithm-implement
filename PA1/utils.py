import numpy as np
from typing import List
from hw1_knn import KNN

### import other library for realize pass by value
import copy

#assumption data is S,Brach(List[List[int]])
#Branch should be like [ [6,7,7,2,10,8,6], [2,0,6,10,4,0,0]...] which has each row for specifc index brach
# each number means how many instances in label class
#entropy is about the certrainty of this brach we dont need to think about meaning of each class results
#which have first arg as number of instances and second arg as number of instances this brach has
# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    def entropy(branch:List[int]) -> List[float]:
        result:List[float] = []
        b = sum(branch)
        for a in branch:
            if(a/b == 0):
                result.append(0)
            else:
                result.append(-1*(a/b)*np.log2(a/b))
        return result
    
    
    entropy_list:List[List[float]] = []
    sum_of_instances:List[int] = []
    
    for branch in branches:
        entropy_list.append(entropy(branch))
        sum_of_instances.append(sum(branch))
    sum_of_all_cases = sum(sum_of_instances)
    
    result = S
    for entropy_row, num_of_case in zip(entropy_list, sum_of_instances):
        result -= sum(entropy_row) * num_of_case / sum_of_all_cases
    if result < 10**(10):
        result = round(result, 10)

    return result
    #raise NotImplementedError


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree trained based on training data set.
    # X_test: List[List[any]] test data, num_cases*num_attributes
    # y_test: List test labels, num_cases*1

    #     Reduced Error Pruning
    # 0. Split data into training and validation sets.
    # 1. Do until further pruning is harmful:
    # 2. Evaluate impact on validation set of pruning each possible node (plus those below it)
    # 3. Greedily remove the one that most improves validation set accuracy
    # - Produces smallest version of most accurate subtree.
    # - Requires that a lot of data be available.

    #########################
    ### new method
    #########################

    assert len(X_test) == len(y_test)


    #first time init best_performance
    best_performance:float = 1.0
    bset_index = []
    best_index_list:List[List[int]] = []
    

    best_performance, best_index = traverse_tree(decisionTree, decisionTree.root_node, None, X_test, y_test, 1, [])
    ### ensure necessary for pruning
    if cal_accuray(decisionTree,X_test,y_test) <= best_performance:
        print('No necessary for prun')
        return
    

    #print('best_performance is:'+ str(best_performance))
    #print('best index is:'+str(best_index))

    dummy_tree = copy.deepcopy(decisionTree)
    best_index_list.append(best_index)
    dummy_tree.delete(best_index)
    cur_performance, cur_index = traverse_tree(dummy_tree, dummy_tree.root_node, None, X_test, y_test, 1, [])

    while cur_performance < best_performance:
        best_index_list.append(cur_index)
        best_performance = cur_performance
        best_index = cur_index
        dummy_tree.delete(cur_index)
        cur_performance, cur_index = traverse_tree(dummy_tree, dummy_tree.root_node, None, X_test, y_test, 1, [])
    
    for delete_traget in best_index_list:
        decisionTree.delete(delete_traget)

    #raise NotImplementedError



# calculate misclassification rate for a specific tree    
def cal_accuray(dummy_tree, X_test, y_test) -> float:

    y_est_test = dummy_tree.predict(X_test)
    assert len(y_est_test) == len(y_test)
    misclass_num = 0

    for a,b in zip(y_est_test, y_test):
        if a != b:
            misclass_num +=1
    
    return misclass_num / len(y_est_test)


#traverse a tree first
def traverse_tree(model_tree,cur_node,last_node, X_test, y_test, best_performance, best_index):

    best_performance = best_performance
    best_index = best_index
    #print('current node is:' + str(cur_node.index))

    #create a copy tree and prun current node
    #copy_tree = (lambda x: (lambda self: x))(model_tree)
    copy_tree = copy.deepcopy(model_tree)
    copy_tree.delete(cur_node.index)


    if cal_accuray(copy_tree, X_test, y_test) < best_performance:
        best_performance = cal_accuray(copy_tree, X_test, y_test)
        best_index = cur_node.index

    elif cal_accuray(copy_tree, X_test, y_test) == best_performance and len(best_index) > len(cur_node.index):
        best_performance = cal_accuray(copy_tree, X_test, y_test)
        best_index = cur_node.index
        


    # base case: this function end
    if not cur_node.splittable:
        # print('best performance:'+ str(best_performance))
        # print('best node index:'+ str(best_index))
        return best_performance, best_index
        

    if cur_node.splittable:
        for a in cur_node.children:
            best_performance,best_index = traverse_tree(model_tree,a,cur_node,X_test,y_test,best_performance,best_index)
    
    return best_performance, best_index
      


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])


    if node.splittable:
        # #TEST ONLY    
        # # if node.dim_split is None: 
        # #     print(node.features[0])
        # print('index of node is:'+ str(node.index))

        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)

    return 2*sum(a*b for a,b in zip(real_labels,predicted_labels))/(sum(real_labels)+sum(predicted_labels))
    

def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    return np.sqrt(sum([(a - b)**2 for a,b in zip(point1,point2)]))    


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return sum(a*b for a,b in zip(point1,point2))


def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    return  -1*np.exp(-1/2*(euclidean_distance(point1,point2)**2))


def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    return 1-inner_product_distance(point1,point2)/(np.sqrt(sum(a**2 for a in point1)) * np.sqrt(sum(b**2 for b in point2)))


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    
    #In this part, you should try different distance function you implemented in part 1.1, and find the best k.
    #Use k range from 1 to 30 and increment by 2. We will use f1-score to compare different models.
    
    #Note: When there is a tie, chose model based on the following priorities:
    #Then check distance function [euclidean > gaussian > inner_prod > cosine_dist];
    #If they have same distance fuction, choose model who have a less k.
    
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    
    # raise NotImplementedError
    
    distance_funcs_pri = {
        'euclidean': 4,
        'gaussian': 3,
        'inner_prod': 2,
        'cosine_dist': 1,
    }
    
    best_model = None
    best_model_perf = None
    best_model_func = None
    
    for dis_name,dis_func in distance_funcs.items():
        
        for k in range(1,min(len(Xtrain) - 1 ,30),2):
        
            cur_model = KNN(k,dis_func)
            cur_model.train(Xtrain,ytrain)
            model_predict = cur_model.predict(Xval)
            cur_model_perf = f1_score(yval, model_predict)
            #print('model performance is:' + str(best_model_perf))
            #print('model current k is:' + str(cur_model.k))
            
            #TODO: check wether F1 score is better with higger         
            if best_model is None or best_model_perf < cur_model_perf:
                best_model = cur_model
                best_model_perf = cur_model_perf
                best_model_func = dis_name

            elif best_model_perf == cur_model_perf:

                if distance_funcs_pri[best_model_func] < distance_funcs_pri[dis_name]:
                    best_model = cur_model
                    best_model_perf = cur_model_perf
                    best_model_func = dis_name
                
                elif distance_funcs_pri[best_model_func] == distance_funcs_pri[dis_name] and best_model.k > cur_model.k:
                    best_model = cur_model
                    best_model_perf = cur_model_perf
                    best_model_func = dis_name
                
                
            

    return best_model, best_model.k, best_model_func


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    distance_funcs_pri = {
        'euclidean': 4,
        'gaussian': 3,
        'inner_prod': 2,
        'cosine_dist': 1,
    }

    normalizaion_pri = {
        'min_max_scale':2,
        'normalize':1,
    }
    
    best_model = None
    best_model_perf = None
    best_model_func = None
    best_model_scaler = None

    for norm_class in scaling_classes:
        scaler = scaling_classes[norm_class]()
        Xtrain_scaled = scaler(Xtrain)
        Xval_scaled = scaler(Xval)

        for dis_name,dis_func in distance_funcs.items():
        
            for k in range(1,min(len(Xtrain) - 1 ,30),2):
        
                cur_model = KNN(k,dis_func)
                cur_model.train(Xtrain_scaled,ytrain)
                model_predict = cur_model.predict(Xval_scaled)
                cur_model_perf = f1_score(yval, model_predict)

                if best_model is None or best_model_perf < cur_model_perf: 
                    best_model = cur_model
                    best_model_perf = cur_model_perf
                    best_model_func = dis_name
                    best_model_scaler = norm_class

                elif best_model_perf == cur_model_perf:
                    if normalizaion_pri[norm_class] > normalizaion_pri[best_model_scaler]:
                        best_model = cur_model
                        best_model_perf = cur_model_perf
                        best_model_func = dis_name
                        best_model_scaler = norm_class

                    elif normalizaion_pri[norm_class] == normalizaion_pri[best_model_scaler] and distance_funcs_pri[best_model_func] < distance_funcs_pri[dis_name]:
                        best_model = cur_model
                        best_model_perf = cur_model_perf
                        best_model_func = dis_name
                        best_model_scaler = norm_class
                        
                    elif normalizaion_pri[norm_class] == normalizaion_pri[best_model_scaler] and distance_funcs_pri[best_model_func] == distance_funcs_pri[dis_name] and best_model.k > cur_model.k:
                        best_model = cur_model
                        best_model_perf = cur_model_perf
                        best_model_func = dis_name
                        best_model_scaler = norm_class
    
    return best_model, best_model.k, best_model_func, best_model_scaler



class NormalizationScaler:
    def __init__(self):
        pass

    #TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        results:List[List[float]] = []
        for single_data in features:
            single:List[float] = []
            for a in single_data:
                if(inner_product_distance(single_data,single_data) != 0):
                   single.append(a / np.sqrt(inner_product_distance(single_data,single_data)))
                else:
                    single.append(a)
            results.append(single)
        return results



class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """
    
    def __init__(self):
        #for a,b in sample:
        #    results.append([(a - min(left))/(max(left)-min(left)), (b-min(right))/(max(right)-min(right))])
        self.first_time = True
        self.scaling:List[List[float]] = []

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        
        rotated = list(zip(*reversed(features)))
            

        
        if(self.first_time):
            for index in range(len(rotated)):
                self.scaling.append([min(rotated[index]),max(rotated[index])])
                self.first_time = False
        
        results = []
        for single_point in features:
            result = []
            for index, item in enumerate(single_point, start=0):   # default is zero
                if(self.scaling[index][1] == self.scaling[index][0]):
                    result.append(item)
                else:
                    result.append((item - self.scaling[index][0])/(self.scaling[index][1]-self.scaling[index][0]))
            results.append(result)
        return results
