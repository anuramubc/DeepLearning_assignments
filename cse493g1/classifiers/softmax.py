from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
      denominator = 0.0
      
      #score for each training image in X
      scores = X[i].dot(W)      
      #correct_class_score = exp(score_of_correct_class)  
      correct_class_score = np.exp(scores[y[i]])
      for j in range(num_classes):
        #denominator is sum of exp(score_of_each_class)
        denominator += np.exp(scores[j])
      
      #loss = sum of li, where li = -log(correct_class_score/denominator)
      loss += -np.log(correct_class_score/denominator)

      for j in range(num_classes):  
        if j == y[i]:
          #for the correct class: find the probability, p_syi = e^syi /sum(e^sj)
          p_syi = correct_class_score/denominator
          #for the correct class: derivative is X(p_syi-1)
          dW[:,j] +=  (p_syi-1)*X[i]
 
        else:
          #for all other classes: find their respective probability, p_sj = e^sj/sum(e^sj)
          p_sj = np.exp(scores[j])/denominator
          #for all other classes: derivative is p_sj(X)
          dW[:,j] +=  (p_sj)*X[i]

    #Average loss  
    loss /= num_train  

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #total dW = average of dW over number of training examples + derivative of regularization term wrt W (2*reg*W)
    dW = dW/num_train + 2*reg*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #scores of each training image over 10 classes
    scores = X.dot(W)

    #raising each term in scores matrix to an exponent
    exp_s = np.exp(scores)

    #demoniator for each row: sum of all elements columnwise
    denominator = np.sum(exp_s, axis = 1)

    #probability table
    prob_table = exp_s/denominator[:,np.newaxis]
  
    #sum of individual loss from ith training example
    loss_i = np.sum(-np.log(prob_table[np.arange(num_train),y]))
    
    #total loss with regularization
    loss = loss_i/num_train + reg * np.sum(W * W)
    
    #gradient 
    #step1: prob of the correct class = ps_yi-1
    prob_table[np.arange(num_train),y] -=1 

    #step2 : dot product of X and probability table
    dW = X.T.dot(prob_table)

    #avg gradient + regularization loss
    dW = dW/num_train + 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW



