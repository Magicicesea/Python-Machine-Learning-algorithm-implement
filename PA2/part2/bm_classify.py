import numpy as np
import copy 

####TEST ONLY
#import timeit


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logisti

    N, D = X.shape
    assert len(np.unique(y)) == 2

    
    #transform y label from 0/1 to -1/1 now
    copy_y = np.array([list(map(lambda x: -1 if x == 0 else 1,y))]).T
    #print(copy_y)

    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    # 51s no imporvement
    # step_fin = step_size / N

    if loss == "perceptron":

        #print(np.matmul(np.zeros((1,N)),copy_y))
        assert np.matmul(np.ones((1,N)),copy_y) != N

        # repeate loss function value store it and use for
        # can be sure is that value should be a N*1 array
        # because we will determine whether that point is mis classcified

        for itr in range(max_iterations):
            #start_time = timeit.default_timer()
            
            loss_array = np.sum(np.multiply(copy_y, np.add(np.matmul(X,np.array([w]).T),b*np.ones((N,1)))),axis=1)
    
            tmp = list(map(lambda x: 1 if x <= 0 else 0,loss_array))
            loss_result = np.array([tmp]).T

            tmp = np.multiply(-1*loss_result,np.multiply(copy_y,X))
            
            w_nxt = np.add(w,-1*step_size/N * np.sum(tmp,axis=0))

            tmp = np.multiply(-1*loss_result,copy_y)
            b_nxt = b - step_size/N * np.sum(tmp)



            w = w_nxt
            b = b_nxt
            #print('each iter take time:' + str(timeit.default_timer() - start_time))
            #print('b is:'+str(b))
            


        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        # w = np.zeros(D)
        # b = 0
        ############################################
        

    elif loss == "logistic":

        #start_time = timeit.default_timer()

        xy_product = np.multiply(copy_y,X)
        #print(xy_product.shape)
        for itr in range(max_iterations):
            
            #start_time = timeit.default_timer()

            tmp = np.array([w]).T
            z = np.add(np.matmul(xy_product,tmp),b*copy_y)
            #print('time for z:'+str(timeit.default_timer() - start_time))

            #start_time = timeit.default_timer()
            logistic_value = sigmoid(-z)
            w_nxt = np.add(w,step_size / N * np.sum(np.multiply(logistic_value,xy_product),axis=0))
            b_nxt = b + step_size / N * np.sum(np.multiply(logistic_value,copy_y))
            #print('time for w,b:'+str(timeit.default_timer() - start_time))


            w = w_nxt
            b = b_nxt


        ############################################
        # TODO 2 : Edit this if part               #
        # #          Compute w and b here            #
        # w = np.zeros(D)
        # b = 0
        ############################################
        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    print('w is:')
    print(w)
    print('b is:')
    print(b)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    #value = z
    ############################################

    return 1 / (1 + np.exp(-z))

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    #print(np.array([w]).T.shape)

    if loss == "perceptron":

        predict_value = np.add(np.matmul(X,np.array([w]).T),b*np.ones((N,1)))
        tmp = list(map(lambda x: 0 if x <= 0 else 1,predict_value))
        preds = np.array([tmp]).T.flatten()


        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        #preds = np.zeros(N)
        ############################################
        

    elif loss == "logistic":

        predict_value = sigmoid(np.add(np.matmul(X,np.array([w]).T),b*np.ones((N,1))))
        #print(sigmoid(predict_value))
        tmp = list(map(lambda x: 1 if x > 0.5 else 0,predict_value))
        preds = np.array([tmp]).T.flatten()
        #print(preds)

        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        #preds = np.zeros(N)
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        
        
        # #print(np.array([b]).T)
        
        # w_with_b = np.concatenate((w,np.array([b]).T),axis=1)
        # X_with_b = np.concatenate((X,np.ones((N,1))),axis=1)

        # for itr in range(max_iterations):


        #     index = np.random.choice(N,1)
        #     xn = X_with_b[index]
        #     yn = y[index]

        #     xn_wt = np.matmul(xn,w_with_b.T)
            
        #     exp_list = np.exp(xn_wt[0])
        #     sigma_exp = np.sum(exp_list)
            
        #     p_list = exp_list / sigma_exp
        #     p_list[yn[0]] = p_list[yn[0]] - 1


        #     tmp = np.array([p_list]).T
        #     w_with_b = np.add(w_with_b,-1 * step_size * np.matmul(tmp,xn))
        #     #b_nxt = b - step_size / N * np.sum(tmp)

        

        # b = w_with_b[:,D]
        # w = np.delete(w_with_b,-1,axis=1)
        # #print(b)

        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################
        

    elif gd_type == "gd":
        
        # # transform y to one-hot vector

        # y_onehot = []
        # for a in y:
        #     tmp = np.zeros(C)
        #     tmp[a] = 1
        #     y_onehot.append(list(tmp))
        # y_onehot = np.array(y_onehot).T

        # X_norm = []
        # for idx in range(N):
        #     row = X[idx,:]
        #     X_norm.append(np.subtract(row,max(row)))
        # X_norm = np.array(X_norm)
        # #print(X_norm.shape)

        # for itr in range(max_iterations):

        #     index_list = np.random.choice(N,N)
        #     x_random = X[index_list]
        #     y_random_onehot = y_onehot.T[index_list].T


        #     xn_wt = np.matmul(x_random,w.T)
            
        #     exp_list = np.exp(xn_wt)
        #     sigma_exp = np.sum(exp_list,axis=0)
            
        #     p_list = np.divide(exp_list, sigma_exp)

        #     p_list = np.add(p_list,-1*y_random_onehot.T)

        #     w = np.add(w, -1 * step_size / N * np.matmul(p_list.T,x_random))


        #print(y_random_onehot)

        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    # preds = np.zeros(N)
    ############################################


    tmp = np.matmul(w,X.T)
    pred_mat = np.add(tmp,np.matmul(np.array([b]).T,np.ones((1,N))))
    preds = np.array(np.argmax(pred_mat,axis=0))

    assert preds.shape == (N,)
    return preds




        