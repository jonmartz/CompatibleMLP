# A simple Tensorflow 2 layer dense network example
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcessClassifier


class MLP:

    def __init__(self, dataset_path, target_col, train_fraction, training_epochs,
                 n_neurons_in_h1, n_neurons_in_h2, learning_rate, diss_weight, data_subset_ratio, model_before_update):

        df = pd.read_csv(dataset_path)
        # self.dict = defaultdict(LabelEncoder)
        # df = df.apply(lambda x: self.dict[x.name].fit_transform(x))
        Y = df.pop(target_col)
        X = df.loc[:]

        # data = datasets.load_breast_cancer()
        # X = data.data[:, :]
        # Y = data.target

        # min max scale and binarize the target labels
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X,Y)
        label = LabelBinarizer()
        Y = label.fit_transform(Y)

        # shuffle dataset
        rows = len(X)
        idx = np.random.randint(X.shape[0], size=rows)
        X = X[idx]
        Y = Y[idx]

        # extract dataset subset
        X = X[:int(rows*data_subset_ratio)]
        Y = Y[:int(rows*data_subset_ratio)]

        train_stop = int(len(X) * train_fraction)

        X_ = X[:train_stop]
        Y_ = Y[:train_stop]

        X_t = X[train_stop:]
        Y_t = Y[train_stop:]

        # # plot the first 3 PCA dimensions of the sampled data
        # fig = plt.figure(1, figsize=(8, 6))
        # ax = Axes3D(fig, elev=-150, azim=110)
        # X_reduced = PCA(n_components=3).fit_transform(X_)
        # ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y_.ravel(),
        #            cmap=plt.cm.Set1, edgecolor='k', s=40)
        # ax.set_title("First three PCA directions")
        # ax.set_xlabel("1st eigenvector")
        # ax.w_xaxis.set_ticklabels([])
        # ax.set_ylabel("2nd eigenvector")
        # ax.w_yaxis.set_ticklabels([])
        # ax.set_zlabel("3rd eigenvector")
        # ax.w_zaxis.set_ticklabels([])
        # plt.show()

        #############################################
        # create the TF neural net
        n_features = len(X[0])
        labels_dim = 1

        # these placeholders serve as our input tensors
        x = tf.placeholder(tf.float32, [None, n_features], name='input')
        y = tf.placeholder(tf.float32, [None, labels_dim], name='labels')

        # input tensor for our reference model predictions
        y_g = tf.placeholder(tf.float32, [None, labels_dim], name='labels')

        # TF Variables are our neural net parameter tensors, we initialize them to random (gaussian) values in
        # Layer1. Variables are allowed to be persistent across training epochs and updatable bt TF operations
        self.W1 = tf.Variable(tf.truncated_normal([n_features, n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)),
                         name='weights1')
        self.b1 = tf.Variable(tf.truncated_normal([n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)), name='biases1')

        # note the output tensor of the 1st layer is the activation applied to a
        # linear transform of the layer 1 parameter tensors
        # the matmul operation calculates the dot product between the tensors
        y1 = tf.sigmoid((tf.matmul(x, self.W1) + self.b1), name='activationLayer1')

        # network parameters(weights and biases) are set and initialized (Layer2)
        self.W2 = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2], mean=0, stddev=1),
                         name='weights2')
        self.b2 = tf.Variable(tf.random_normal([n_neurons_in_h2], mean=0, stddev=1), name='biases2')
        # activation function(sigmoid)
        y2 = tf.sigmoid((tf.matmul(y1, self.W2) + self.b2), name='activationLayer2')

        # output layer weights and biases
        self.Wo = tf.Variable(tf.random_normal([n_neurons_in_h2, labels_dim], mean=0, stddev=1 ),
                         name='weightsOut')
        self.bo = tf.Variable(tf.random_normal([labels_dim], mean=0, stddev=1), name='biasesOut')

        # the sigmoid (binary softmax) activation is absorbed into TF's sigmoid_cross_entropy_with_logits loss
        logits = tf.matmul(y2, self.Wo) + self.bo
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = logits)

        # tap a separate output that applies softmax activation to the output layer
        # for training accuracy readout
        a = tf.nn.sigmoid(logits, name='activationOutputLayer')

        # compare predicted value from network with the expected value/target
        correct_prediction = tf.equal(tf.round(a), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

        if model_before_update is not None:

            # dissonance calculation
            correct_old = tf.cast(tf.equal(tf.round(y_g), y), tf.float32)
            diss = correct_old * loss
            loss = loss + diss_weight * diss

            # compatibility calculation
            correct_new = tf.cast(tf.equal(tf.round(a), y), tf.float32)
            compatibility = tf.reduce_sum(correct_old * correct_new) / tf.reduce_sum(correct_old)

        # optimizer used to compute gradient of loss and apply the parameter updates.
        # the train_step object returned is ran by a TF Session to train the net

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        #############################################

        # ***NOTE global_variables_initializer() must be called before creating a tf.Session()!***
        init_op = tf.global_variables_initializer()

        # create a session for training and feedforward (prediction). Sessions are TF's way to run
        # feed data to placeholders and variables, obtain outputs and update neural net parameters

        with tf.Session() as sess:
            # ***initialization of all variables... NOTE this must be done before running any further sessions!***
            sess.run(init_op)

            # training loop over the number of epochs
            batch_size = 50
            batches = int(len(X_) / batch_size)

            if model_before_update is None:  # without compatibility
                print("------------"
                      "\nTRAINING h1:\ntrain fraction = "+str(100*train_fraction)+"%\n")
                for epoch in range(training_epochs):
                    losses = 0
                    accs = 0
                    for j in range(batches):
                        idx = np.random.randint(X_.shape[0], size=batch_size)
                        X_b = X_[idx]
                        Y_b = Y_[idx]

                        # train the network, note the dictionary of inputs and labels
                        sess.run(train_step, feed_dict={x: X_b, y: Y_b})
                        # feedforwad the same data and labels, but grab the accuracy and loss as outputs
                        acc, l, soft_max_a = sess.run([accuracy, loss, a], feed_dict={x: X_b, y: Y_b})

                        losses = losses + np.sum(l)
                        accs = accs + np.sum(acc)
                    print("%.3d" % (epoch+1) + "/" + str(training_epochs), "\tTRAIN: avg loss = %.4f" % (losses / batches),
                          "\tacc = ", "%.4f" % (accs / batches))

                    # test on the holdout set
                    acc, l, soft_max_a = sess.run([accuracy, loss, a], feed_dict={x: X_t, y: Y_t})
                    print("\t\tTEST:  loss = %.4f" % np.sum(l), "\tacc = %.4f" % acc)
                    print()

            else:  # with compatibility
                print("-------------------------------"
                      "\nTRAINING h2 COMPATIBLE WITH h1:\ntrain fraction = "+
                      str(100*train_fraction)+"%, diss weight = "+str(diss_weight)+"\n")
                for epoch in range(training_epochs):
                    losses = 0
                    diss_losses = 0
                    accs = 0
                    for j in range(batches):
                        idx = np.random.randint(X_.shape[0], size=batch_size)
                        X_b = X_[idx]
                        Y_b = Y_[idx]

                        # get the old model predictions... and slice only the positive class probabilities
                        Y_g = model_before_update.predict_probabilities(X_b)
                        # Y_g = Y_g[:, 1].reshape((-1, 1))
                        # Y_g = model_before_update.predict_proba(X_b)[:, 1].reshape((-1, 1))

                        # train the network, note the dictionary of inputs and labels
                        sess.run(train_step, feed_dict={x: X_b, y: Y_b, y_g: Y_g})
                        # feedforwad the same data and labels, but grab the accuracy and loss as outputs
                        acc, l, soft_max_a, l_2 = sess.run([accuracy, loss, a, diss],
                                                           feed_dict={x: X_b, y: Y_b, y_g: Y_g})
                        losses = losses + np.sum(l)
                        accs = accs + np.sum(acc)
                        diss_losses = diss_losses + np.sum(l_2)

                    print("%.3d" % (epoch+1) + "/" + str(training_epochs), "\tTRAIN: avg loss = %.4f" % (losses / batches),
                          "\tacc = ", "%.4f" % (accs / batches))

                    # test on the holdout set
                    Y_g = model_before_update.predict_probabilities(X_t)
                    # Y_g = model_before_update.predict_proba(X_t)[:, 1].reshape((-1, 1))

                    acc, l, soft_max_a, com = sess.run([accuracy, loss, a, compatibility],
                                                       feed_dict={x: X_t, y: Y_t, y_g: Y_g})
                    print("\t\tTEST:  loss = %.4f" % np.sum(l), "\tacc = %.4f" % acc)
                    print("\t\t       diss = %.4f" % diss_losses, "\t\tcom = %.4f " % com)
                    print()

    def predict_probabilities(self, x):
        x = tf.cast(x, tf.float32)
        y1 = tf.sigmoid((tf.matmul(x, self.W1) + self.b1), name='activationLayer1')
        y2 = tf.sigmoid((tf.matmul(y1, self.W2) + self.b2), name='activationLayer2')
        logits = tf.matmul(y2, self.Wo) + self.bo
        return tf.nn.sigmoid(logits, name='activationOutputLayer').eval()


dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\creditRiskAssessment\\heloc_dataset_v1.csv'
target_col = 'RiskPerformance'
h1 = MLP(dataset_path, target_col, 0.05, 20, 10, 10, 0.05, 0.5, 1.0, None)
h2 = MLP(dataset_path, target_col, 0.2, 20, 10, 10, 0.05, 0.5, 1.0, h1)