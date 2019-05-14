"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

# test need to combine Q1.2 

###### Q1.1 ######
def mean_square_error(w, X, y):

    

    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    
    # assuming y could be multi class question
    
    if len(X.shape) == 1:
      xr = 1
      xc = X.shape[0]
    else:
      xr,xc = X.shape

    if len(y.shape) == 1:
      yr = 1
      yc = y.shape[0]
    else:
      yr,yc = y.shape

    if len(w.shape) == 1:
      wr = 1
      wc = w.shape[0]
    else:
      wr,wc = w.shape

    assert wr*wc*yr*yc*xr*wc != 0

    # assuming num_samples, D, _ does not equal to any one other
    # if so we cannot figure out them in function

    if xc != wr:
      #find who is the wrong one or both
      
      #assume x is wrong w is right
      if xr == wr:
        xr,xc = xc,xr
        X.transpose()
      #assume x is right w is wrong
      elif xc == wc:
        wr,wc = wc,wr
        w.transpose()
      #assume x and w are both wrong
      elif xr == wc:
        xr,xc = xc,xr
        X.transpose()
        wr,wc = wc,wr
        w.transpose()
    

    # if D is correct for both X,w
    if xc == wr:
      pred_y = np.matmul(X,w)
      if xr == yr:
        return np.mean(np.power(pred_y - y,2))
        #means wc and yc 
      elif xr != yr:  
        y.transpose()
        yr,yc = yc,yr
        return np.mean(np.power(pred_y - y,2))
    
        
    #err = None
    #return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################	

  tmp_mult_x = np.matmul(np.linalg.inv(np.matmul(X.transpose(),X)),X.transpose())

  if len(tmp_mult_x.shape) == 1:
    xr = 1
    xc = tmp_mult_x.shape[0]
  else:
    xr,xc = tmp_mult_x.shape

  if len(y.shape) == 1:
    yr = 1
    yc = y.shape[0]
  else:
    yr,yc = y.shape

  # if X, y both right
  if xc == yr:
    return np.matmul(tmp_mult_x,y)
  
  # if X is not right and y right
  elif xr == yr:
    tmp_mult_x.transpose()

  # if X is right and y is wrong
  elif xc == yc:
    y.transpose()
  # if X and y is both wrong
  elif xr == yc:
    tmp_mult_x.transpose()
    y.transpose()
  
  return np.matmul(tmp_mult_x,y)

  # w = None
  # return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################

    if len(y.shape) == 1 :
      y = np.array([y])

    if len(X.shape) == 1:
      xr = 1
      xc = X.shape[0]
    else:
      xr,xc = X.shape

    if len(y.shape) == 1:
      yr = 1
      yc = y.shape[0]
    else:
      yr,yc = y.shape

    # y is wrong
    if xr == yc:
      yr,yc = yc,yr
      y = y.T
    # x is wrong
    elif xc == yr:
      xr,xc = xc,xr
      X.transpose()
    # y and x both wrong
    elif xc == yc:
      yr,yc = yc,yr
      y = y.transpose()
      xr,xc = xc,xr
      X.transpose()
    
    tmp_product = np.matmul(X.transpose(),X)

    # determin whether x is invertable
    eigval, _ = np.linalg.eig(tmp_product)
    min_eig_val = min(np.absolute(eigval))
    while min_eig_val < 10**-5:
      ### this matrix is invertable
      tmp_product = np.add(tmp_product, 10**-1 * np.identity(len(tmp_product)))
      eigval, _ = np.linalg.eig(tmp_product)
      min_eig_val = min(np.absolute(eigval))
    
    


    w = np.matmul(np.matmul(np.linalg.inv(tmp_product),X.transpose()),y).flatten()
    return w


    # w = None
    # return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################	

    if len(y.shape) == 1 :
      y = np.array([y])

    if len(X.shape) == 1:
      xr = 1
      xc = X.shape[0]
    else:
      xr,xc = X.shape

    if len(y.shape) == 1:
      yr = 1
      yc = y.shape[0]
    else:
      yr,yc = y.shape

    # y is wrong
    if xr == yc:
      yr,yc = yc,yr
      y = y.T
    # x is wrong
    elif xc == yr:
      xr,xc = xc,xr
      X.transpose()
    # y and x both wrong
    elif xc == yc:
      yr,yc = yc,yr
      y = y.transpose()
      xr,xc = xc,xr
      X.transpose()
    
    tmp_product = np.matmul(X.transpose(),X)

    tmp_product = np.add(tmp_product, lambd * np.identity(len(tmp_product)))

    w = np.matmul(np.matmul(np.linalg.inv(tmp_product),X.transpose()),y).flatten()
    return w 
   


###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    # bestlambda = None
    # return bestlambda
    lambda_range = range(-19,20,1)
    best_performance = None
    best_lambd = None

    for lambd in lambda_range:
      w = regularized_linear_regression(Xtrain,ytrain,10**lambd)
      mse = mean_square_error(w,Xval,yval)
      if best_performance is None or best_performance > mse :
        best_performance = mse
        best_lambd = lambd
    
    # print(best_performance)
    
    return 10**best_lambd
    

    



###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################

    result = None
    for cur_power in range(1,power+1,1):
      tmp_x = np.power(X,cur_power)
      if cur_power == 1:
        result = tmp_x
      else:
        result = np.insert(result,[len(result[0])],tmp_x, axis=1)
    return result


