import numpy as np
import utils as Util
from typing import List

class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()
        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred

    # return taregt index 
    def delete(self,target_index:List[int]) ->List[int]:
        cur_node = self.root_node

        if len(target_index) == 1:
            cur_node.prun_node()
            return False

        #find point first
        for a in target_index[1:]:
            if a >= len(cur_node.children):
                return False
            cur_node = cur_node.children[a]
        cur_node.prun_node()
        return cur_node.index


    ###TEST ONLY
    #return a specific node
    def search(self,target_index):
        cur_node = self.root_node

        #find point first
        for a in target_index[1:]:
            if a >= len(cur_node.children):
                return False
            cur_node = cur_node.children[a]
        return cur_node





class TreeNode(object):
    def __init__(self, features, labels, num_cls,index:List[int] = [0]):
        # features: List[List[any]], labels: List[int], num_cls: int

        # parametes given with initialization
        # self.cls_max is for specific label with most counts
        # self.splittable is whether still able for further split

        # following is parameters need to be applied
        # self.dim_split is index of feature to be split
        # self.feature_uniq_split are candidate features for further split

        ###
        # Note: self.feature_uniq_split must be sorted in any cases

        ###
        # for further convinence have index for delection function
        # print(index)
        self.index:List[int] = index
        #print('init index is:'+str(self.index))

        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        
        elif len(self.features[0]) == 0:
            self.splittable = False
            # print(self.features)
            # print(self.labels)
            # print(self.cls_max)

        else:
            self.splittable = True
        

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split:List[int] = []  # the possible unique values of the feature to be split


    #TODO: try to split current node

    # In the TreeNode class, the features variable means all the points in current TreeNode,
    # and the labels variable means the corresponding labels for all data.
    # The children variable is a list of TreeNode after split the current node based on the best attributs.
    # This should be a recursive process that once we call the split function, the TreeNode will keep spliting 
    # untill we get the whole tree structure.
    def split(self):

        
        if not self.splittable:
            # this case for only one class (self.cls_max)
            # set all split as default and return self.cls_max as result
            return

        elif len(self.features[0]) == 0 and self.num_cls != 0:
            # this case for no more features available
            # choose majority of classes as result self.cls_max
            return 
        
        elif len(self.features[0]) == 0 and self.num_cls == 0:
            # this case return majority of classes with parent node
            # !!! Consider when predicating parent result should be hold
            return 
        
        
        #TODO: produce specifc feature branch result    
        candidate_value_list:List[List[any]] = []
        tmp = np.sort(np.array(self.features).transpose())
        for row in tmp:
            # candidate_value_list is transpose feature matrix with deduplication
            candidate_value_list.append(np.unique(row))


        Entropy_for_Features:List[float] = []

        
        # candidate_feature is each feature row with its index in feature matrix
        
        
        for index,candidate_feature in enumerate(candidate_value_list, start=0):

            tmp_branches_data:List[List[int]] = []
            for current_value in candidate_feature:
                class_dic = dict()
                for class_label in sorted(np.unique(self.labels)):
                    class_dic[class_label] = 0
                
                #choose instances with specific feature value and return as branch set,
                #this branch set should have deleted specific locaiton feature
                for label_index, row in enumerate(self.features, start=0):
                    if row[index] == current_value:
                        class_dic[self.labels[label_index]] +=1
                #entropy for a specifc feature
                tmp_branch_data:List[int] = []
                for _,value in class_dic.items():
                    tmp_branch_data.append(value)
                #need normalization for tmp_brach_data
                tmp_branches_data.append(tmp_branch_data)
            
            #directly append entropy for each attribute    
            #Entropy_for_Features.append(-1*Util.Information_Gain(0,tmp_branches_data))
            #follow instruction produce S
            #Entropy_for_Features.append(-1*Util.Information_Gain(self.entropy_root(),tmp_branches_data))
            #print('tmp_branches_data_for_each'+str(tmp_branches_data))
            
            Entropy_for_Features.append(Util.Information_Gain(self.entropy_root(),tmp_branches_data))
        
        #print('Entropy for features' + str(Entropy_for_Features))

        #get a entropy list in Entropy_for_Features:List[float]
        #consider when have same entropy value and how to compare
        # find index of all max entropy
        candidate_features:List[int] = []

        for index, entropy in enumerate(Entropy_for_Features, start=0):
            if entropy == max(Entropy_for_Features):
                candidate_features.append(index)
        #print('value of Entropys is:' + str(Entropy_for_Features))
        # if only one maximum entropy
        if len(candidate_features) == 1:
            self.dim_split = candidate_features[0]
        # more than one maximum entropy
        # init: transpose features matrix for picking up data line
        transpose_features = np.array(self.features).transpose()
        # storage for best candidate
        if len(candidate_features) > 1:
            #print("============")
            
            #print("equal entropy features: "+ str(candidate_features))
            best_candidate_index = len(transpose_features) + 1
            best_unique_number = 0
            for candidate_feature_index in candidate_features:
                # oringinal based on possible kinds of values for [2, 4, 5, 7] is 4
                unique_feature_number = len(np.unique(transpose_features[candidate_feature_index]))

                # try based on range of values for [2, 4, 5 ,7] should be 5
                #unique_feature_number = max(transpose_features[candidate_feature_index]) - min(transpose_features[candidate_feature_index])
                # print("all feature values:"+ str(transpose_features[candidate_feature_index]))
                # if unique_feature_number_1 != unique_feature_number:
                    
                #     print("range of feature value:" + str(unique_feature_number))
                #     print("new range of feature:" + str(unique_feature_number_1))
                if unique_feature_number > best_unique_number:
                    best_candidate_index = candidate_feature_index
                    best_unique_number = unique_feature_number
                elif (unique_feature_number == best_unique_number) and (best_candidate_index > candidate_feature_index):
                    best_candidate_index = candidate_feature_index
                    best_unique_number = unique_feature_number
            self.dim_split = best_candidate_index

        #print("final choice feature:" + str(self.dim_split))
        # dimension has been chosen
        
        #####
        # TODO: check variable
        # put candidate unique values into self.feature_uniq_split
        # self.feature_uniq_split = np.unique(transpose_features[self.dim_split])
        #####
        
        # initialize treenode and put in self.children
        # features, labels, num_cls are required parameters
        # pick up data row with specific value
        # feature value from min to max
        feature_values = sorted(np.unique(transpose_features[self.dim_split]))
        for value_index,cur_value in enumerate(feature_values,start=0):
            children_features:List[any] = []
            children_labels:List[int] = []
            for index,feature_row in enumerate(self.features, start=0):
                if feature_row[self.dim_split] == cur_value:
                    
                    #features with specific value has been taken out
                    #labels for that position
                    tmp = list(feature_row)
                    tmp.pop(self.dim_split)
                    children_features.append(tmp)
                    children_labels.append(self.labels[index])
            # num_cls for new node
            children_num_cls = len(np.unique(children_labels))
            # new index should be added based on parent one and its index
            new_index = self.index.copy()
            new_index.append(value_index)
            #print(value_index)

            new_instance = TreeNode(children_features, children_labels, children_num_cls,new_index)

            new_instance.split()
            
            self.children.append(new_instance)
            self.feature_uniq_split.append(cur_value)
        
        # each node has increasing order of feature values
        #print('feature values should be increasing order: ' + str(self.feature_uniq_split))



    # TODO: predict the branch or the class
    def predict(self, feature) -> int:
        # feature: List[any]
        # return: int predicated class
        tmp_feature = feature.copy()
        if type(tmp_feature) == np.ndarray :
            tmp_feature = tmp_feature.tolist()
        
        # reach leaf
        if len(self.children) == 0:
            return self.cls_max
        
        for index, boundary in enumerate(self.feature_uniq_split,start=0):
           
            if feature[self.dim_split] == boundary:
                
                #if feature[self.dim_split] != boundary:
                    #print('boundary:'+ str(boundary) + ' feature value:'+ str(feature[self.dim_split]))
                    #print('feature list' + str(self.feature_uniq_split))
                    
                    ## test acc: 0.5936952714535902
                    #return self.cls_max

                # hit the branch, return branch result but features has to be removed on specific column
                #print('dimension to be splited:'+str(self.dim_split))
                #print('oringal data:' + str(feature))
                tmp_feature.pop(self.dim_split)
                #print('reduced data'+str(tmp_feature))
                return self.children[index].predict(tmp_feature)
        
        return self.cls_max
        
        #raise NotImplementedError


    #helper funciton for calculating root entropy
    def entropy_root(self) -> float:
        #result = 0
        sum = 0
        for label in np.unique(self.labels):
            sum += -1*self.labels.count(label)/len(self.labels)*np.log2(self.labels.count(label)/len(self.labels))
        return sum

    # def Information_Gain(self, features:List[any]) -> float:
    
    #Objective: claim this node is not splitable further and request prediction function use cls_max as prediction result
    def prun_node(self):
        # assuming node has been initialized and no more new node is allowed to produce in this stage
        self.splittable = False
        self.children = []
        self.dim_split = None
        self.feature_uniq_split = []

