from builtins import range
from matplotlib.pyplot import margins
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
    #for i in range(1,2):
        scores = X[i].dot(W)        
        correct_class_score = scores[y[i]]       
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1           
            if margin > 0:
                loss += margin
                # for each class j != y[i] whose margin >0, the derivative of loss with respect to the weight of that classj (Wj)
                # is X[i]
                dW[:,j] += X[i]
                # for each class j != y[i] whose margin >0, the derivative of loss with respect to the weight of the correct class yi (W_yi) 
                # is -X[i]
                dW[:, y[i]] -= X[i]
   

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #total dW = average of dW over number of training examples + derivative of regularization term wrt W (2*reg*W)
    dW = dW/num_train + 2*reg*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #vectorized operation:
    num_train = X.shape[0]

    #scores of each training image over 10 classes
    scores_v = X.dot(W)

    #scores of correct class
    correct_class_score_v = scores_v[np.arange(num_train),y]

    #SVM loss
    margin_v = scores_v - correct_class_score_v[:,np.newaxis] + 1

    #changing margins of correct classes in each traing example to 0
    margin_v[np.arange(num_train),y] = 0

    #replace all negative margins to 0
    margin_v[margin_v<0] = 0
    
    #Li = loss for each training sample = sum over all columns in margins array
    Loss_i = np.sum(margin_v, axis = 1)
    #loss_v = average loss across all traing examples
    loss = np.mean(Loss_i)

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #location of elements in margin_v that are non zero, since gradient is valid only for margings >0
    idx = np.zeros(margin_v.shape)
    idx[margin_v >0] = 1

    #at the index of the correct label, replace 0 with sum number of other classes with margins >0 
    idx[np.arange(num_train),y] = -np.sum(idx, axis = 1)

    #matrix multiplication of X^T and idx 
    dW = (X.T).dot(idx)
    dW = dW/num_train + 2*reg*W

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
