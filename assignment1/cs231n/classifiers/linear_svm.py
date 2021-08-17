from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    数据有D维，有C个分类
    Inputs:
    - W: A numpy array of shape (D, C) containing weights. 权重矩阵
    - X: A numpy array of shape (N, D) containing a minibatch of data. N个D维向量
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.N个样本对应的标签
    - reg: (float) regularization strength正则化强度

    Returns a tuple of:
    - loss as single float  一个float的loss
    - gradient with respect to weights W; an array of same shape as W D维C分类的梯度dw
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero
    # 初始化梯度为dW

    # compute the loss and the gradient
    num_classes = W.shape[1] # 数据的类别C
    num_train = X.shape[0] # 训练集数据个数N
    loss = 0.0
    # 开始计算lossfunc  hinge loss
    for i in range(num_train): # 遍历每一个样本i
        scores = X[i].dot(W)  # 线性 点乘 样本i(D,)乘W(D,C) 得到(C,),为每一类的分数
        correct_class_score = scores[y[i]] # 当前样本对应的label类别的分数
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0: # 不然loss为0 ji
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    # 遍历所有样本后loss 已经是sum后的值了，进行一个平均值的取
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2*reg*W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    # 与其先计算损失然后再计算导数，不如在计算损失的同时计算导数更简单。
    # 因此，您可能需要修改上面的一些代码来计算梯度。
    # 已经修改上方loss计算代码，增加梯度计算
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
        Inputs:
    - W: A numpy array of shape (D, C) containing weights. 权重矩阵
    - X: A numpy array of shape (N, D) containing a minibatch of data. N个D维向量
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.N个样本对应的标签
    - reg: (float) regularization strength正则化强度
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    # 计算loss
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]  # 训练集数据个数N
    scores = X.dot(W)  # N,C  N个样本各自有C个类的分数
    # N,C 每个样本在每个类别上的loss
    margin = scores - scores[range(0,num_train), y].reshape(-1,1) + 1
    margin = (margin > 0) * margin # 小于0 的不取
    margin[range(num_train), y] = 0  # 对应类别设置为0
    loss += margin.sum() / num_train
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
    # dW D*C
    counts = (margin > 0).astype(int) # 小于0为0，大于0为1 n*c
    # 有多少个j不为0，就要加多少个 -y[i]
    counts[range(num_train), y] = - np.sum(counts, axis=1) # axis=1 按行方向求和
    # X.T D,N   counts N,C  =>D,C
    dW += np.dot(X.T, counts) / num_train + 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
