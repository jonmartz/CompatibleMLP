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

    def __init__(self, X, Y, train_fraction, train_epochs, batch_size, layer_sizes, learning_rate,
                 diss_weight=None, old_model=None, dissonance_type=None, make_h1_subset=None):

        start_time = int(round(time.time() * 1000))

        # ------------ #
        # PREPARE DATA #
        # ------------ #

        # shuffle indexes to cover train and test sets
        if old_model is None or not make_h1_subset:
            shuffled_indexes = np.random.randint(len(X), size=len(X))
            train_stop = int(len(X) * train_fraction)
            self.train_indexes = shuffled_indexes[:train_stop]
            self.test_indexes = shuffled_indexes[train_stop:]
        else:  # make the old train set to be a subset of the new train set
            # shuffled = np.random.randint(len(old_model.test_indexes), size=len(old_model.test_indexes))
            # shuffled_test_indexes = old_model.test_indexes[shuffled]

            # todo: trying to eliminate randomness here
            shuffled_test_indexes = old_model.test_indexes
            test_stop = int(len(X) * train_fraction - len(old_model.train_indexes))
            self.train_indexes = np.concatenate((old_model.train_indexes, shuffled_test_indexes[:test_stop]))
            self.test_indexes = shuffled_test_indexes[test_stop:]

        # assign train and test subsets
        X_train = X[self.train_indexes]
        Y_train = Y[self.train_indexes]
        X_test = X[self.test_indexes]
        Y_test = Y[self.test_indexes]

        # print("X_train = " + str(X_train))
        # print("Y_train = " + str(Y_train))
        # print("X_test = " + str(X_test))
        # print("Y_test = " + str(Y_test))

        # ------------ #
        # CREATE MODEL #
        # ------------ #

        n_features = len(X[0])
        labels_dim = 1

        # these placeholders serve as the input tensors
        x = tf.placeholder(tf.float32, [None, n_features], name='input')
        y = tf.placeholder(tf.float32, [None, labels_dim], name='labels')

        # input tensor for the base model predictions
        y_old_labels = tf.placeholder(tf.float32, [None, labels_dim], name='old_labels')
        y_old_correct = tf.placeholder(tf.float32, [None, labels_dim], name='old_corrects')

        # set initial weights
        initial_weights = []
        initial_biases = []
        if old_model is None or not make_h1_subset:
            initial_weights += [tf.truncated_normal([n_features, layer_sizes[0]], mean=0, stddev=1 / np.sqrt(n_features))]
            initial_biases += [tf.truncated_normal([layer_sizes[0]], mean=0, stddev=1 / np.sqrt(n_features))]

            for i in range(len(layer_sizes) - 1):
                initial_weights += [tf.random_normal([layer_sizes[i], layer_sizes[i + 1]], mean=0, stddev=1)]
                initial_biases += [tf.random_normal([layer_sizes[i + 1]], mean=0, stddev=1)]

            initial_weights += [tf.random_normal([layer_sizes[-1], labels_dim], mean=0, stddev=1)]
            initial_biases += [tf.random_normal([labels_dim], mean=0, stddev=1)]
        else:
            # todo: trying to eliminate randomness here
            for layer_weights in old_model.final_weights:
                initial_weights += [tf.convert_to_tensor(layer_weights)]
            for layer_biases in old_model.final_biases:
                initial_biases += [tf.convert_to_tensor(layer_biases)]

        # first layer
        weights = [tf.Variable(initial_weights[0], name='weights_In')]
        biases = [tf.Variable(initial_biases[0], name='biases_In')]
        activations = [tf.sigmoid((tf.matmul(x, weights[0]) + biases[0]), name='activation_In')]

        # second layer
        for i in range(len(initial_weights) - 1)[1:]:
            weights += [tf.Variable(initial_weights[i], name='weights_'+str(i))]
            biases += [tf.Variable(initial_biases[i], name='biases_'+str(i))]
            activations += [tf.sigmoid((tf.matmul(activations[i-1], weights[i]) + biases[i]), name='activation_'+str(i))]

        # output layer
        weights += [tf.Variable(initial_weights[-1], name='weights_Out')]
        biases += [tf.Variable(initial_biases[-1], name='biases_Out')]
        logits = tf.matmul(activations[-1], weights[-1]) + biases[-1]
        output = tf.nn.sigmoid(logits, name='activation_Out')

        # accuracy tensors
        correct_prediction = tf.equal(tf.round(output), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

        # log loss
        log_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

        # dissonance calculation
        if old_model is None:
            pass
        elif dissonance_type == "D":
            dissonance = y_old_correct * tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        elif dissonance_type == "D'":
            dissonance = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_old_labels, logits=logits)
        elif dissonance_type == "D''":
            dissonance = y_old_correct * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_old_labels, logits=logits)
        else:
            raise Exception("invalid dissonance type. The valid input strings are: D, D' and D''")

        # loss computation
        if old_model is None:
            loss = log_loss
        else:
            loss = (1-diss_weight)*log_loss + diss_weight*dissonance

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

            if old_model is None:  # without compatibility
                print("\nTRAINING h1:\ntrain fraction = " + str(int(100 * train_fraction)) + "%\n")

                for epoch in range(train_epochs):
                    losses = 0
                    accs = 0
                    for batch in range(batches + 1):
                        # shuffled_indexes = np.random.randint(X_train.shape[0], size=batch_size)
                        # X_batch = X_train[shuffled_indexes]
                        # Y_batch = Y_train[shuffled_indexes]

                        # todo: trying to eliminate randomness here
                        batch_start = batch * batch_size
                        if batch_start == X_train.shape[0]:
                            continue  # in case the len of train set is a multiple of batch size
                        batch_end = min((batch + 1) * batch_size, X_train.shape[0])
                        X_batch = X_train[batch_start:batch_end]
                        Y_batch = Y_train[batch_start:batch_end]

                        # train the model, and then get the accuracy and loss from model
                        sess.run(train_step, feed_dict={x: X_batch, y: Y_batch})
                        acc, lss, out = sess.run([accuracy, loss, output], feed_dict={x: X_batch, y: Y_batch})
                        losses = losses + np.sum(lss)
                        accs = accs + np.sum(acc)

                    # test the model
                    acc, lss, out = sess.run([accuracy, loss, output], feed_dict={x: X_test, y: Y_test})

                    # preds = tf.round(out).eval()
                    # zeroes = 0
                    # ones = 0
                    # for pred in preds:
                    #     if pred == 0.0:
                    #         zeroes += 1
                    #     else:
                    #         ones += 1
                    # print("ones = "+str(ones)+"/"+str(ones+zeroes))

                    print(str(epoch + 1) + "/" + str(train_epochs) + "\ttrain acc = %.4f" % (accs / batches)
                          + ", test acc = %.4f" % acc)

                    self.accuracy = acc

                # save weights and biases
                self.final_weights = []
                self.final_biases = []
                for layer_weights in weights:
                    self.final_weights += [layer_weights.eval()]
                for layer_biases in biases:
                    self.final_biases += [layer_biases.eval()]

                print("test acc = %.4f" % acc)

            else:  # with compatibility
                print("\nTRAINING h2 COMPATIBLE WITH h1:\ntrain fraction = " +
                      str(int(100 * train_fraction)) + "%, diss weight = " + str(diss_weight)
                      + ", diss type = " + str(dissonance_type) + "\n")

                # get the old model predictions
                Y_train_old_probabilities = old_model.predict_probabilities(X_train)
                Y_train_old_labels = tf.round(Y_train_old_probabilities).eval()
                Y_train_old_correct = tf.cast(tf.equal(Y_train_old_labels, Y_train), tf.float32).eval()
                Y_test_old_probabilities = old_model.predict_probabilities(X_test)
                Y_test_old_labels = tf.round(Y_test_old_probabilities).eval()
                Y_test_old_correct = tf.cast(tf.equal(Y_test_old_labels, Y_test), tf.float32).eval()

                # first = True
                for epoch in range(train_epochs):
                    losses = 0
                    diss_losses = 0
                    accs = 0
                    for batch in range(batches):
                        # shuffled_indexes = np.random.randint(X_train.shape[0], size=batch_size)
                        # X_batch = X_train[shuffled_indexes]
                        # Y_batch = Y_train[shuffled_indexes]
                        # Y_batch_old_labels = Y_train_old_labels[shuffled_indexes]
                        # Y_batch_old_correct = Y_train_old_correct[shuffled_indexes]

                        # todo: trying to eliminate randomness here
                        batch_start = batch * batch_size
                        if batch_start == X_train.shape[0]:
                            continue  # in case the len of train set is a multiple of batch size
                        batch_end = min((batch + 1) * batch_size, X_train.shape[0])
                        X_batch = X_train[batch_start:batch_end]
                        Y_batch = Y_train[batch_start:batch_end]
                        Y_batch_old_labels = Y_train_old_labels[batch_start:batch_end]
                        Y_batch_old_correct = Y_train_old_correct[batch_start:batch_end]

                        # train the new model, and then get the accuracy and loss from it
                        sess.run(train_step, feed_dict={x: X_batch, y: Y_batch,
                                                        y_old_labels: Y_batch_old_labels,
                                                        y_old_correct: Y_batch_old_correct})
                        acc, lss, out, diss = sess.run([accuracy, loss, output, dissonance],
                                                       feed_dict={x: X_batch, y: Y_batch,
                                                                  y_old_labels: Y_batch_old_labels,
                                                                  y_old_correct: Y_batch_old_correct})
                        # if losses == 0 and epoch == 0:
                        #     print(diss_type + ": "+str(diss))

                        losses = losses + np.sum(lss)
                        accs = accs + np.sum(acc)
                        diss_losses = diss_losses + np.sum(diss)

                    # test the new model
                    acc, lss, out, com = sess.run([accuracy, loss, output, compatibility],
                                                  feed_dict={x: X_test, y: Y_test,
                                                             y_old_labels: Y_test_old_labels,
                                                             y_old_correct: Y_test_old_correct})

                    # if first:
                    #     first = False
                    #     print("acc = "+str(acc))
                    #
                    # preds = tf.round(out).eval()
                    # zeroes = 0
                    # ones = 0
                    # for pred in preds:
                    #     if pred == 0.0:
                    #         zeroes += 1
                    #     else:
                    #         ones += 1
                    # if ones > 0:
                    #     print("ones = " + str(ones) + ", epoch = " + str(epoch) + ", acc = "+str(acc))

                    # print(str(epoch + 1) + "/" + str(train_epochs) + "\ttrain acc = %.4f" % (accs / batches)
                    #       + ", test acc = %.4f" % acc + ", com = %.4f" % com)

                    self.accuracy = acc
                    self.compatibility = com

                print("FINISHED:\ttest accuracy = %.4f" % self.accuracy + ", compatibility = %.4f" % self.compatibility)

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
        y = tf.sigmoid((tf.matmul(x, self.final_weights[0]) + self.final_biases[0])).eval()
        for i in range(len(self.final_weights)-1)[1:]:
            y = tf.sigmoid((tf.matmul(y, self.final_weights[i]) + self.final_biases[i])).eval()
        logits = (tf.matmul(y, self.final_weights[-1]) + self.final_biases[-1]).eval()
        return tf.nn.sigmoid(logits, name='activationOutputLayer').eval()


# ------------------ #
# todo: MODIFY THESE #
# ------------------ #

# Data-set paths
dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\creditRiskAssessment\\heloc_dataset_v1.csv'
results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\results\\creditRiskAssessment.csv"
target_col = 'RiskPerformance'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\hospitalMortalityPrediction\\ADMISSIONS_encoded.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\results\\hospitalMortalityPrediction.csv"
# target_col = 'HOSPITAL_EXPIRE_FLAG'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\recividismPrediction\\compas-scores-two-years_encoded.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\results\\recividismPrediction.csv"
# target_col = 'is_recid'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\fraudDetection\\train_transaction_short_encoded.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\results\\fraudDetection.csv"
# target_col = 'isFraud'

# Data fractions
dataset_fraction = 0.5
# dataset_fraction = 0.2
h1_train_fraction = 0.02
h2_train_fraction = 0.2

# Dissonance types
# diss_types = ["D", "D'", "D''"]
# diss_types = ["D", "D'"]
diss_types = ["D"]

# Dissonance weights in [0,100] as percentages
diss_weights = range(21)
# diss_weights = [0]
factor = 4  # multiplies the diss by this factor
repetitions = 1

# END OF MODIFYING SECTION #

# get rows and labels
df = pd.read_csv(dataset_path)
Y = df.pop(target_col)
X = df.loc[:]

# extract dataset subset
rows = len(X)
X = X[:int(rows * dataset_fraction)]
Y = Y[:int(rows * dataset_fraction)]

# min max scale and binarize the target labels
scaler = MinMaxScaler()
X = scaler.fit_transform(X, Y)
label = LabelBinarizer()
Y = label.fit_transform(Y)

# compute base model:
h1 = MLP(X, Y, h1_train_fraction, 50, 50, [10, 5], 0.02)
with open(results_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["compatibility", "h1 accuracy", "h2 accuracy (D)", "h2 accuracy (D')", "h2 accuracy (D'')",
                     "dissonance weight", "h1 train fraction = " + str(h1_train_fraction), "h2 train fraction = " + str(h2_train_fraction)])

# train compatible models:
iteration = 0
for i in diss_weights:
    for j in range(repetitions):
        offset = 0
        for diss_type in diss_types:
            iteration += 1
            print("-------\n"+str(iteration)+"/"+str(len(diss_weights) * repetitions * len(diss_types))+"\n-------")
            diss_weight = factor * i / 100.0
            tf.reset_default_graph()
            h2 = MLP(X, Y, h2_train_fraction, 200, 50, [10, 5], 0.02, diss_weight, h1, diss_type, True)
            with open(results_path, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                row = [str(h2.compatibility), str(h1.accuracy)]
                for k in diss_types:
                    row += [""]
                row += [str(diss_weight)]
                row[2 + offset] = str(h2.accuracy)
                writer.writerow(row)
            offset += 1
