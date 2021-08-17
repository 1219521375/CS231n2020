from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.
    In other words, the network has the following architecture:
    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """
    """
    两层完全连接的神经网络。网络的输入维度为N、H的一个隐层维数，并对C类进行分类。
    我们使用softmax损失函数和L2正则化对网络进行训练权重矩阵。
    在第一次完全恢复后，网络使用了ReLU非线性连接层。
    
    换句话说，网络具有以下架构：
    输入-完全连接层-ReLU-完全连接层-softmax
    第二个完全连接层的输出是每个类的分数。
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        """
        初始化模型。权重初始化为小的随机值,偏差(bias)初始化为零。
        权重和偏差存储在变量self.params，它是具有以下键的字典：
        W1：第一层权重； 具有形状（D，H）
        b1：第一层偏差； 形状为（H，）
        W2：第二层权重； 具有形状（H，C）
        b2：第二层偏差； 形状为（C，）
        输入：
        - input_size：输入数据的尺寸D。
        - hidden_size：隐藏层中神经元H的数量。
        - output_size：类C的数量。
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)  # D,H
        self.params['b1'] = np.zeros(hidden_size)  # H,
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)  # H,C
        self.params['b2'] = np.zeros(output_size)  # C,

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        """
        计算两层全连接神经网络的损失loss和梯度
        输入：
        - X：输入形状（N，D）的数据。每个X[i]是一个训练样本。
        - y: 训练标签向量。(N,) y[i]是X[i]的标签，每个y[i]是范围0<=y[i]<C的整数。
        此参数是可选的；如果它没有被传递，那么我们只返回分数，如果传递了，那么我们返回损失和梯度。
        - reg: 正则化强度。
        
        返回值：
        如果y为None，则返回形状为（N，C）的分数矩阵，其中scores[i，c]为输入X [i]上类C的分数。
        如果y不为None，则返回一个元组tuple：
        - loss：这批训练样本的损失（数据损失和规范化损失）。
        - grad：将参数名称映射到这些参数相对于损失函数的梯度的字典；具有与self.params相同的键。
        """
        # Unpack variables from the params dictionary
        # 从params字典解包变量
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass 计算前向传播
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # 第一层 输入X，参数W1,b1 ，输出z2
        z1 = np.dot(X, W1) + b1  # N,H + H, ==> N,H  全连接的输出
        z2 = np.maximum(0, z1)  # RELU的输出  N,H
        # 第二层 输入z2，参数W2,b2，输出softmax之后的z3
        z3 = np.dot(z2, W2) + b2  # N,C + C, ==> N,C  全连接的输出 连接softmax
        scores = z3
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        scores -= np.max(scores, axis=1).reshape(-1, 1)  # 防止数值爆炸 将score中的值平移到最大值为0：
        # loss softmax的loss化简为log - log 后一个log与e抵消
        softmax_out = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1, 1)
        loss = -np.sum(np.log(softmax_out[range(N), list(y)]))
        loss = loss / N + reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        # 计算反向传递，计算权重和偏差的导数。 将结果存储在grads词典中。
        # 例如，grads ['W1']应该将梯度存储在W1上，并且是大小相同的矩阵
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # dW2 == dsoftout/dW2 == 第二层softmax的输出对于W2的梯度
        dW2 = softmax_out.copy()
        dW2[range(N), y] -= 1
        dW2 /= N  # N,C   # softmax的loss对输入softmax的输入分数求导
        grads['W2'] = np.dot(z2.T, dW2) + 2 * reg * W2  # H,C
        grads['b2'] = np.sum(dW2, axis=0)  # C,

        # 上一层传回的梯度对第一层输出的梯度 dz2/dw1
        dz2 = dW2.dot(W2.T)  # N,H  求出传回的梯度
        dz2_relu = (z2 > 0) * dz2  # z2>0就是取输入中大于0的部分 N,H
        grads['W1'] = np.dot(X.T,dz2_relu) + 2 * reg * W1  # D,N * N,H ==> D,H
        grads['b1'] = np.sum(dz2_relu,axis=0)  # H,
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx]
            y_batch = y[idx]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        h = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
        scores = h.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
