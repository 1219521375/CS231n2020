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
    # compute the loss and the gradient
    num_classes = W.shape[1]  # 数据的类别C
    num_train = X.shape[0]  # 训练集数据个数N

    # 开始计算lossfunc  hinge loss
    for i in range(num_train):  # 遍历每一个样本i
        scores = X[i].dot(W)  # 线性 点乘 样本i(D,)乘W(D,C) 得到(C,),为当前样本每一类的分数
        scores -= np.max(scores)  # 防止梯度爆炸 ，这里可以不用，下面向量化必须用
        scores_exp = np.exp(scores)  # 求e
        scores_exp_sum = scores_exp.sum()  # 当前样本的分数e和
        correct_class_score = scores_exp[y[i]]  # 当前样本对应的label类别的分数
        loss -= np.log(correct_class_score / scores_exp_sum)
        dW[:, y[i]] -= X[i]
        for j in range(num_classes):
            dW[:, j] += X[i] * scores_exp[j] / scores_exp_sum

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # compute the loss and the gradient
    num_classes = W.shape[1]  # 数据的类别C
    num_train = X.shape[0]  # 训练集数据个数N
    scores = X.dot(W)  # N,C
    scores -= np.max(scores, axis=1).reshape(-1, 1)  # 防止数值爆炸 将score中的值平移到最大值为0：
    # N,C / N,1
    softmax_output = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1, 1)
    loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))
    loss /= num_train
    loss += reg * np.sum(W * W)

    # dW D,C
    counts = np.exp(scores) / (np.sum(np.exp(scores), axis=1).reshape(-1,1)) # N,C
    counts[range(num_train),y] -= 1
    dW = np.dot(X.T,counts)   # D,N * N,C ==> D,C
    dW = dW / num_train + 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
