import csv
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer


class MLP:
    """
    Class implementing a Multi Layered Perceptron, capable of training using the log loss + dissonance
    to be able to produce compatible updates.
    """

    def __init__(self, dataset_path, target_col, train_fraction, training_epochs, batch_size,
                 n_neurons_in_h1, n_neurons_in_h2, learning_rate, diss_weight, data_subset_ratio, model_before_update):

        # ------------ #
        # PREPARE DATA #
        # ------------ #

        # get rows and labels
        df = pd.read_csv(dataset_path)
        Y = df.pop(target_col)
        X = df.loc[:]

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

        # separate train and test subsets
        train_stop = int(len(X) * train_fraction)
        X_train = X[:train_stop]
        Y_train = Y[:train_stop]
        X_test = X[train_stop:]
        Y_test = Y[train_stop:]

        # ------------ #
        # CREATE MODEL #
        # ------------ #

        n_features = len(X[0])
        labels_dim = 1

        # these placeholders serve as the input tensors
        x = tf.placeholder(tf.float32, [None, n_features], name='input')
        y = tf.placeholder(tf.float32, [None, labels_dim], name='labels')

        # input tensor for the base model predictions
        y_old_correct = tf.placeholder(tf.float32, [None, labels_dim], name='labels')

        # first layer
        self.W1 = tf.Variable(tf.truncated_normal([n_features, n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)), name='weights1')
        self.b1 = tf.Variable(tf.truncated_normal([n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)), name='biases1')
        y1 = tf.sigmoid((tf.matmul(x, self.W1) + self.b1), name='activationLayer1')

        # second layer
        self.W2 = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2], mean=0, stddev=1), name='weights2')
        self.b2 = tf.Variable(tf.random_normal([n_neurons_in_h2], mean=0, stddev=1), name='biases2')
        y2 = tf.sigmoid((tf.matmul(y1, self.W2) + self.b2), name='activationLayer2')

        # output layer
        self.Wo = tf.Variable(tf.random_normal([n_neurons_in_h2, labels_dim], mean=0, stddev=1 ), name='weightsOut')
        self.bo = tf.Variable(tf.random_normal([labels_dim], mean=0, stddev=1), name='biasesOut')
        logits = tf.matmul(y2, self.Wo) + self.bo
        output = tf.nn.sigmoid(logits, name='activationOutputLayer')

        # accuracy tensors
        correct_prediction = tf.equal(tf.round(output), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

        # loss tensor
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        if model_before_update is not None:

            # dissonance calculation
            # correct_old = tf.cast(tf.equal(tf.round(y_old), y), tf.float32)
            # dissonance = correct_old * loss
            dissonance = y_old_correct * loss
            loss = loss + diss_weight * dissonance

            # compatibility calculation
            correct_new = tf.cast(tf.equal(tf.round(output), y), tf.float32)
            compatibility = tf.reduce_sum(y_old_correct * correct_new) / tf.reduce_sum(y_old_correct)

        # ----------- #
        # TRAIN MODEL #
        # ----------- #

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            batches = int(len(X_train) / batch_size)
            acc = 0
            com = 0
            self.accuracy = 0
            self.compatibility = 0

            if model_before_update is None:  # without compatibility
                print("------------"
                      "\nTRAINING h1:\ntrain fraction = "+str(100*train_fraction)+"%\n")

                for epoch in range(training_epochs):
                    losses = 0
                    accs = 0
                    for j in range(batches):
                        idx = np.random.randint(X_train.shape[0], size=batch_size)
                        X_batch = X_train[idx]
                        Y_batch = Y_train[idx]

                        # train the model, and then get the accuracy and loss from model
                        sess.run(train_step, feed_dict={x: X_batch, y: Y_batch})
                        acc, lss, out = sess.run([accuracy, loss, output], feed_dict={x: X_batch, y: Y_batch})
                        losses = losses + np.sum(lss)
                        accs = accs + np.sum(acc)

                    print("%.3d" % (epoch+1) + "/" + str(training_epochs), "\tTRAIN: avg loss = %.4f" % (losses / batches),
                          "\tacc = %.4f" % (accs / batches))

                    # test the model
                    acc, lss, out = sess.run([accuracy, loss, output], feed_dict={x: X_test, y: Y_test})
                    print("\t\t\tTEST:  loss = %.4f" % np.sum(lss), "\tacc = %.4f" % acc)
                    print()

                    if acc > self.accuracy:
                        self.accuracy = acc

            else:  # with compatibility
                print("-------------------------------"
                      "\nTRAINING h2 COMPATIBLE WITH h1:\ntrain fraction = "+
                      str(100*train_fraction)+"%, diss weight = "+str(diss_weight)+"\n")

                # get the old model predictions
                Y_train_old = model_before_update.predict_probabilities(X_train)
                Y_train_old_correct = tf.cast(tf.equal(tf.round(Y_train_old), Y_train), tf.float32).eval()

                for epoch in range(training_epochs):
                    losses = 0
                    diss_losses = 0
                    accs = 0
                    for j in range(batches):
                        idx = np.random.randint(X_train.shape[0], size=batch_size)
                        X_batch = X_train[idx]
                        Y_batch = Y_train[idx]
                        Y_batch_old_correct = Y_train_old_correct[idx]

                        # train the new model, and then get the accuracy and loss from it
                        sess.run(train_step, feed_dict={x: X_batch, y: Y_batch, y_old_correct: Y_batch_old_correct})
                        acc, lss, out, diss = sess.run([accuracy, loss, output, dissonance],
                                                       feed_dict={x: X_batch, y: Y_batch, y_old_correct: Y_batch_old_correct})
                        losses = losses + np.sum(lss)
                        accs = accs + np.sum(acc)
                        diss_losses = diss_losses + np.sum(diss)

                    print("%.3d" % (epoch+1) + "/" + str(training_epochs), "\tTRAIN: avg loss = %.4f" % (losses / batches),
                          "\tacc = %.4f" % (accs / batches))

                    # test the new model
                    Y_old = model_before_update.predict_probabilities(X_test)
                    Y_old_correct = tf.cast(tf.equal(tf.round(Y_old), Y_test), tf.float32).eval()
                    acc, lss, out, com = sess.run([accuracy, loss, output, compatibility],
                                                       feed_dict={x: X_test, y: Y_test, y_old_correct: Y_old_correct})

                    print("\t\tTEST:  loss = %.4f" % np.sum(lss), "\tacc = %.4f" % acc)
                    print("\t\t       diss = %.4f" % diss_losses, "\tcom = %.4f " % com)
                    print()

                    # prioritize high accuracy over high compatibility
                    if acc > self.accuracy:
                        self.accuracy = acc
                        self.compatibility = com


    def predict_probabilities(self, x):
        """
        predict the labels for dataset x
        :param x: dataset to predict labels of
        :return: numpy array with the probability for each label
        """
        x = tf.cast(x, tf.float32)
        y1 = tf.sigmoid((tf.matmul(x, self.W1) + self.b1), name='activationLayer1')
        y2 = tf.sigmoid((tf.matmul(y1, self.W2) + self.b2), name='activationLayer2')
        logits = tf.matmul(y2, self.Wo) + self.bo
        return tf.nn.sigmoid(logits, name='activationOutputLayer').eval()


# data
dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\creditRiskAssessment\\heloc_dataset_v1.csv'
results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\results\\creditRiskAssessment.csv"
target_col = 'RiskPerformance'

# compute base model:
h1 = MLP(dataset_path, target_col, 0.05, 100, 50, 10, 10, 0.02, 0.5, 1.0, None)
with open(results_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["accuracy", "compatibility", "dissonance weight"])
    writer.writerow([str(h1.accuracy), "(base accuracy)"])

# train compatible models:
for i in range(50, 51):
    diss_weight = i/100.0
    h2 = MLP(dataset_path, target_col, 0.2, 50, 50, 10, 10, 0.02, diss_weight, 1.0, h1)
    with open(results_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([str(h2.accuracy), str(h2.compatibility), str(diss_weight)])
