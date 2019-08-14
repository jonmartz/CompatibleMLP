import csv
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
import time

class MLP:
    """
    Class implementing a Multi Layered Perceptron, capable of training using the log loss + dissonance
    to be able to produce compatible updates.
    """

    def __init__(self, X_original, Y_original, train_fraction, train_epochs, batch_size,
                 layer_1_neurons, layer_2_neurons, learning_rate, diss_weight, data_subset_ratio, model_before_update):

        start_time = int(round(time.time() * 1000))

        # ------------ #
        # PREPARE DATA #
        # ------------ #

        # shuffle dataset
        rows = len(X_original)
        idx = np.random.randint(X_original.shape[0], size=rows)
        X = X_original[idx]
        Y = Y_original[idx]

        # extract dataset subset
        X = X[:int(rows * data_subset_ratio)]
        Y = Y[:int(rows * data_subset_ratio)]

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
        W1 = tf.Variable(
            tf.truncated_normal([n_features, layer_1_neurons], mean=0, stddev=1 / np.sqrt(n_features)), name='weights1')
        b1 = tf.Variable(tf.truncated_normal([layer_1_neurons], mean=0, stddev=1 / np.sqrt(n_features)),
                              name='biases1')
        y1 = tf.sigmoid((tf.matmul(x, W1) + b1), name='activationLayer1')

        # second layer
        W2 = tf.Variable(tf.random_normal([layer_1_neurons, layer_2_neurons], mean=0, stddev=1), name='weights2')
        b2 = tf.Variable(tf.random_normal([layer_2_neurons], mean=0, stddev=1), name='biases2')
        y2 = tf.sigmoid((tf.matmul(y1, W2) + b2), name='activationLayer2')

        # output layer
        Wo = tf.Variable(tf.random_normal([layer_2_neurons, labels_dim], mean=0, stddev=1), name='weightsOut')
        bo = tf.Variable(tf.random_normal([labels_dim], mean=0, stddev=1), name='biasesOut')
        logits = tf.matmul(y2, Wo) + bo
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
            y_new_correct = tf.cast(tf.equal(tf.round(output), y), tf.float32)
            compatibility = tf.reduce_sum(y_old_correct * y_new_correct) / tf.reduce_sum(y_old_correct)

        # prepare training
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        init_op = tf.global_variables_initializer()

        # ----------- #
        # TRAIN MODEL #
        # ----------- #

        with tf.Session() as sess:
            sess.run(init_op)
            batches = int(len(X_train) / batch_size)
            acc = 0
            com = 0
            self.accuracy = 0
            self.compatibility = 0

            if model_before_update is None:  # without compatibility
                print("\nTRAINING h1:\ntrain fraction = " + str(100 * train_fraction) + "%\n")

                for epoch in range(train_epochs):
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

                    # test the model
                    acc, lss, out = sess.run([accuracy, loss, output], feed_dict={x: X_test, y: Y_test})
                    print(str(epoch + 1) + "/" + str(train_epochs) + "\ttrain acc = %.4f" % (accs / batches)
                          + ", test acc = %.4f" % acc)

                    # if acc > self.accuracy:
                    self.accuracy = acc

                # save weights
                self.final_W1 = W1.eval()
                self.final_b1 = b1.eval()
                self.final_W2 = W2.eval()
                self.final_b2 = b2.eval()
                self.final_Wo = Wo.eval()
                self.final_bo = bo.eval()

                print("test acc = %.4f" % acc)

            else:  # with compatibility
                # min_loss = -1
                print("\nTRAINING h2 COMPATIBLE WITH h1:\ntrain fraction = " +
                      str(100 * train_fraction) + "%, diss weight = " + str(diss_weight) + "\n")

                # get the old model predictions
                Y_train_old = model_before_update.predict_probabilities(X_train)
                Y_train_old_correct = tf.cast(tf.equal(tf.round(Y_train_old), Y_train), tf.float32).eval()
                Y_test_old = model_before_update.predict_probabilities(X_test)
                Y_test_old_correct = tf.cast(tf.equal(tf.round(Y_test_old), Y_test), tf.float32).eval()

                for epoch in range(train_epochs):
                    # print("graph version = " + str(tf.get_default_graph().version))
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
                                                       feed_dict={x: X_batch, y: Y_batch,
                                                                  y_old_correct: Y_batch_old_correct})
                        losses = losses + np.sum(lss)
                        accs = accs + np.sum(acc)
                        diss_losses = diss_losses + np.sum(diss)

                    # test the new model
                    acc, lss, out, com = sess.run([accuracy, loss, output, compatibility],
                                                  feed_dict={x: X_test, y: Y_test, y_old_correct: Y_test_old_correct})

                    print(str(epoch + 1) + "/" + str(train_epochs) + "\ttrain acc = %.4f" % (accs / batches)
                          + ", test acc = %.4f" % acc + ", com = %.4f" % com)

                    # prioritize high accuracy over high compatibility
                    # curr_loss = sum(lss)
                    # if min_loss == -1 or curr_loss < min_loss:
                    #     min_loss = curr_loss
                    self.accuracy = acc
                    self.compatibility = com

                print("BEST:\ttest accuracy = %.4f" % self.accuracy + ", compatibility = %.4f" % self.compatibility)

            runtime = str(int((round(time.time() * 1000))-start_time)/1000)
            print("runtime = " + str(runtime) + " secs\n")

    def predict_probabilities(self, x):
        """
        predict the labels for dataset x
        :param x: dataset to predict labels of
        :return: numpy array with the probability for each label
        """
        # print("\nWEIGHTS:\nW1 = "+str(self.final_W1[:3])+"\nb1 = "+str(self.final_b1[:3])
        #       + "\nW2 = " + str(self.final_W2[:3]) + "\nb2 = " + str(self.final_b2[:3])
        #       + "\nWo = " + str(self.final_Wo[:3]) + "\nbo = " + str(self.final_bo[:3]))
        x = tf.cast(x, tf.float32).eval()
        y1 = tf.sigmoid((tf.matmul(x, self.final_W1) + self.final_b1)).eval()
        y2 = tf.sigmoid((tf.matmul(y1, self.final_W2) + self.final_b2)).eval()
        logits = (tf.matmul(y2, self.final_Wo) + self.final_bo).eval()
        return tf.nn.sigmoid(logits, name='activationOutputLayer').eval()


# data
dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\creditRiskAssessment\\heloc_dataset_v1.csv'
results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\results\\creditRiskAssessment.csv"
target_col = 'RiskPerformance'
h1_train_fraction = 0.02
h2_train_fraction = 0.5
dataset_fraction = 0.5

# get rows and labels
df = pd.read_csv(dataset_path)
Y = df.pop(target_col)
X = df.loc[:]

# min max scale and binarize the target labels
scaler = MinMaxScaler()
X = scaler.fit_transform(X, Y)
label = LabelBinarizer()
Y = label.fit_transform(Y)

# compute base model:
h1 = MLP(X, Y, h1_train_fraction, 100, 50, 10, 10, 0.02, None, dataset_fraction, None)
with open(results_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["compatibility", "h1 accuracy", "h2 accuracy", "dissonance weight",
                     "h1 train fraction = " + str(h1_train_fraction), "h2 train fraction = " + str(h2_train_fraction)])

# train compatible models:
diss_weights = 41
repetitions = 3
iteration = 0
for i in range(diss_weights):
# for i in [0,81]:
    for j in range(repetitions):
        iteration += 1
        print("-------\n"+str(iteration)+"/"+str(diss_weights * repetitions)+"\n-------")
        diss_weight = 5*i / 100.0
        tf.reset_default_graph()
        h2 = MLP(X, Y, h2_train_fraction, 100, 50, 10, 10, 0.02, diss_weight, dataset_fraction, h1)
        with open(results_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([str(h2.compatibility), str(h1.accuracy), str(h2.accuracy), str(diss_weight)])

