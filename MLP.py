import csv
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
import time
import matplotlib.pyplot as plt
import os, os.path
from scipy.stats import multivariate_normal

import math


class MLP:
    """
    Class implementing a Multi Layered Perceptron, capable of training using the log loss + dissonance
    to be able to produce compatible updates.
    """

    def __init__(self, X, Y, train_fraction, train_epochs, batch_size, layer_1_neurons, layer_2_neurons, learning_rate,
                 diss_weight=None, old_model=None, dissonance_type=None, make_h1_subset=True, copy_h1_weights=True,
                 history=None, use_history=False, train_start=0, initial_stdev=1, make_train_similar_to_history=False,
                 non_parametric=False):

        start_time = int(round(time.time() * 1000))

        # ------------------------------ #
        # PREPARE TRAINING AND TEST SETS #
        # ------------------------------ #

        # shuffle indexes to cover train and test sets
        if old_model is None or not make_h1_subset:
            # shuffled_indexes = np.random.randint(len(X), size=len(X))
            # train_stop = int(len(X) * train_fraction)
            # self.train_indexes = shuffled_indexes[:train_stop]
            # self.test_indexes = shuffled_indexes[train_stop:]

            # todo: trying to eliminate randomness here
            indexes = range(len(X))
            train_size = int(len(X) * train_fraction)
            self.train_indexes = indexes[train_start*train_size:(train_start+1)*train_size]

            if make_train_similar_to_history:
                similar_indexes = []
                for index in self.train_indexes:
                    if history.likelihood[index] == 1:
                        similar_indexes += [index]
                self.train_indexes = similar_indexes

            self.test_indexes = [x for x in indexes if x not in self.train_indexes]

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

        if history is not None:
            if non_parametric:
                kernels_train = history.kernels[self.train_indexes]
                kernels_test = history.kernels[self.test_indexes]
            else:
                likelihood_train = history.likelihood[self.train_indexes]
                likelihood_test = history.likelihood[self.test_indexes]

        # ------------ #
        # CREATE MODEL #
        # ------------ #

        n_features = len(X[0])
        labels_dim = 1

        # these placeholders serve as the input tensors
        x = tf.placeholder(tf.float32, [None, n_features], name='input')
        y = tf.placeholder(tf.float32, [None, labels_dim], name='labels')
        likelihood = tf.placeholder(tf.float32, [None, labels_dim], name='likelihood')

        # input tensor for the base model predictions
        y_old_probabilities = tf.placeholder(tf.float32, [None, labels_dim], name='old_probabilities')
        y_old_correct = tf.placeholder(tf.float32, [None, labels_dim], name='old_corrects')

        # for non-parametric
        kernels = tf.placeholder(tf.float32, [None, labels_dim], name='hist_kernels')
        hist_y = tf.placeholder(tf.float32, [None, labels_dim], name='hist_labels')
        hist_y_old_logits = tf.placeholder(tf.float32, [None, labels_dim], name='hist_old_logits')
        hist_y_old_correct = tf.placeholder(tf.float32, [None, labels_dim], name='hist_old_corrects')

        # set initial weights
        if old_model is None or not copy_h1_weights:
        # if True:
            w1_initial = tf.truncated_normal([n_features, layer_1_neurons], mean=0, stddev=initial_stdev / np.sqrt(n_features))
            b1_initial = tf.truncated_normal([layer_1_neurons], mean=0, stddev=initial_stdev / np.sqrt(n_features))
            w2_initial = tf.random_normal([layer_1_neurons, layer_2_neurons], mean=0, stddev=initial_stdev)
            b2_initial = tf.random_normal([layer_2_neurons], mean=0, stddev=initial_stdev)
            wo_initial = tf.random_normal([layer_2_neurons, labels_dim], mean=0, stddev=initial_stdev)
            bo_initial = tf.random_normal([labels_dim], mean=0, stddev=initial_stdev)
        else:
            w1_initial = tf.convert_to_tensor(old_model.final_W1)
            b1_initial = tf.convert_to_tensor(old_model.final_b1)
            w2_initial = tf.convert_to_tensor(old_model.final_W2)
            b2_initial = tf.convert_to_tensor(old_model.final_b2)
            wo_initial = tf.convert_to_tensor(old_model.final_Wo)
            bo_initial = tf.convert_to_tensor(old_model.final_bo)

        # first layer
        W1 = tf.Variable(w1_initial, name='weights1')
        b1 = tf.Variable(b1_initial, name='biases1')
        y1 = tf.sigmoid((tf.matmul(x, W1) + b1), name='activationLayer1')

        # second layer
        W2 = tf.Variable(w2_initial, name='weights2')
        b2 = tf.Variable(b2_initial, name='biases2')
        y2 = tf.sigmoid((tf.matmul(y1, W2) + b2), name='activationLayer2')

        # output layer
        Wo = tf.Variable(wo_initial, name='weightsOut')
        bo = tf.Variable(bo_initial, name='biasesOut')
        logits = tf.matmul(y2, Wo) + bo
        output = tf.nn.sigmoid(logits, name='activationOutputLayer')

        # model evaluation tensors
        correct_prediction = tf.equal(tf.round(output), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        y_new_correct = tf.cast(tf.equal(tf.round(output), y), tf.float32)

        # loss computation
        log_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

        # dissonance computation
        if old_model is None:
            loss = log_loss
        else:
            if dissonance_type == "D":
                dissonance = y_old_correct * tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
            elif dissonance_type == "D'":
                dissonance = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_old_probabilities, logits=logits)
            elif dissonance_type == "D''":
                dissonance = y_old_correct * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_old_probabilities,
                                                                                     logits=logits)
            else:
                raise Exception("invalid dissonance type. The valid input strings are: D, D' and D''")

            if history is None:
                print("no history")
                loss = log_loss + diss_weight * dissonance
                compatibility = tf.reduce_sum(y_old_correct * y_new_correct) / tf.reduce_sum(y_old_correct)
            else:
                if not use_history:
                    loss = log_loss + diss_weight * dissonance
                else:
                    # history dissonance
                    if non_parametric:
                        hist_dissonance = hist_y_old_correct * tf.nn.sigmoid_cross_entropy_with_logits(labels=hist_y,
                                                                                                       logits=hist_y_old_logits)
                        # todo: try with and without hist_dissonance
                        kernel_likelihood = np.sum(kernels * hist_dissonance) / np.sum(kernels)
                        # kernel_likelihood = np.sum(kernels) / len(kernels)

                        # todo: try with and without normal dissonance
                        loss = log_loss + diss_weight * kernel_likelihood
                        # loss = log_loss + diss_weight * kernel_likelihood * dissonance
                    else:
                        loss = log_loss + diss_weight * dissonance * likelihood

                compatibility = tf.reduce_sum(y_old_correct * y_new_correct * likelihood) / tf.reduce_sum(y_old_correct * likelihood)
        loss = loss/batch_size

        # prepare training
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) (obsolete)
        train_step = tf.train.AdamOptimizer().minimize(loss)
        init_op = tf.global_variables_initializer()

        # ----------- #
        # TRAIN MODEL #
        # ----------- #

        with tf.Session() as sess:
            sess.run(init_op)
            batches = int(len(X_train) / batch_size)

            if old_model is None:  # without compatibility
                print("\nTRAINING h1:\ntrain fraction = " + str(int(100 * train_fraction)) + "%\n")

                plot_x = []
                plot_losses = []
                plot_auc = []
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
                        # acc, lss, out = sess.run([accuracy, loss, output], feed_dict={x: X_batch, y: Y_batch})
                        # losses = losses + np.sum(lss)
                        # accs = accs + np.sum(acc)

                    # test the model
                    acc, lss, out = sess.run([accuracy, loss, output], feed_dict={x: X_test, y: Y_test})
                    self.auc = sklearn.metrics.roc_auc_score(Y_test, out)

                    # print(str(epoch + 1) + "/" + str(train_epochs) + "\tloss = %.4f" %losses +"\tauc = %.4f" % self.auc)

                #     plot_x += [epoch]
                #     plot_losses += [losses]
                #     plot_auc += [ self.auc]
                #
                # plt.plot(plot_x, plot_losses, 'bs', label='loss')
                # plt.plot(plot_x, plot_auc, 'r^', label='auc')
                # plt.xlabel('epoch')
                # plt.legend(('loss', 'auc'),loc='center left')
                # plt.title('Training')
                # plt.show()
                # plt.clf()

                print("test auc = %.4f" % self.auc)

            else:  # with compatibility

                if use_history:
                    history_string = "USE HISTORY"
                else:
                    history_string = "IGNORE HISTORY"

                print("\nTRAINING h2 COMPATIBLE WITH h1:\ntrain fraction = " +
                      str(int(100 * train_fraction)) + "%, diss weight = " + str(diss_weight)
                      + ", diss type = " + str(dissonance_type) + ", " + history_string + "\n")

                # get the old model predictions
                Y_train_old_probabilities = old_model.predict_probabilities(X_train)
                Y_train_old_labels = tf.round(Y_train_old_probabilities).eval()
                Y_train_old_correct = tf.cast(tf.equal(Y_train_old_labels, Y_train), tf.float32).eval()
                Y_test_old_probabilities = old_model.predict_probabilities(X_test)
                Y_test_old_labels = tf.round(Y_test_old_probabilities).eval()
                Y_test_old_correct = tf.cast(tf.equal(Y_test_old_labels, Y_test), tf.float32).eval()

                hist_Y_old_logits = old_model.predict_probabilities(history.instances, True)
                hist_Y_old_probabilities = tf.nn.sigmoid(hist_Y_old_logits, name='hist_probabilities').eval()
                hist_Y_old_labels = tf.round(hist_Y_old_probabilities).eval()
                hist_Y_old_correct = tf.cast(tf.equal(hist_Y_old_labels, history.labels), tf.float32).eval()

                for epoch in range(train_epochs):
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
                        Y_batch_old_probabilities = Y_train_old_probabilities[batch_start:batch_end]
                        Y_batch_old_correct = Y_train_old_correct[batch_start:batch_end]

                        # train the new model, and then get the accuracy and loss from it
                        if history is None:
                            sess.run(train_step, feed_dict={x: X_batch, y: Y_batch,
                                                            y_old_probabilities: Y_batch_old_probabilities,
                                                            y_old_correct: Y_batch_old_correct})
                        else:
                            if non_parametric:
                                kernels_batch = kernels_train[batch_start:batch_end]
                                sess.run(train_step, feed_dict={x: X_batch, y: Y_batch,
                                                                y_old_probabilities: Y_batch_old_probabilities,
                                                                y_old_correct: Y_batch_old_correct,
                                                                hist_y_old_logits: hist_Y_old_logits,
                                                                hist_y_old_correct: hist_Y_old_correct,
                                                                kernels: kernels_batch})
                            else:
                                likelihood_batch = likelihood_train[batch_start:batch_end]
                                sess.run(train_step, feed_dict={x: X_batch, y: Y_batch,
                                                            y_old_probabilities: Y_batch_old_probabilities,
                                                            y_old_correct: Y_batch_old_correct,
                                                            likelihood: likelihood_batch})
                # test the new model
                if history is None:
                    out, com = sess.run([output, compatibility],
                                        feed_dict={x: X_test, y: Y_test,
                                                   y_old_probabilities: Y_test_old_probabilities,
                                                   y_old_correct: Y_test_old_correct})
                else:
                    if non_parametric:
                        out, com, log_lss, diss, new_correct = sess.run(
                            [output, compatibility, log_loss, hist_dissonance, y_new_correct],
                            feed_dict={x: X_test, y: Y_test,
                                       y_old_probabilities: Y_test_old_probabilities,
                                       y_old_correct: Y_test_old_correct,
                                       hist_y_old_logits: hist_Y_old_logits,
                                       hist_y_old_correct: hist_Y_old_correct,
                                       kernels: kernels_test})
                    else:
                        out, com, log_lss, diss, new_correct = sess.run([output, compatibility, log_loss, dissonance, y_new_correct],
                                            feed_dict={x: X_test, y: Y_test,
                                                       y_old_probabilities: Y_test_old_probabilities,
                                                       y_old_correct: Y_test_old_correct,
                                                       likelihood: likelihood_test})

                self.compatibility = com
                self.auc = sklearn.metrics.roc_auc_score(Y_test, out)
                self.new_correct = new_correct
                self.old_correct = Y_test_old_correct

                print("FINISHED:\ttest auc = %.4f" % self.auc + ", compatibility = %.4f" % self.compatibility)
                # print("log loss = "+str(np.sum(log_lss)))
                # print("dissonance = "+str(np.sum(diss)))

            # save weights
            self.final_W1 = W1.eval()
            self.final_b1 = b1.eval()
            self.final_W2 = W2.eval()
            self.final_b2 = b2.eval()
            self.final_Wo = Wo.eval()
            self.final_bo = bo.eval()

            runtime = str(int((round(time.time() * 1000))-start_time)/1000)
            print("runtime = " + str(runtime) + " secs\n")

    def predict_probabilities(self, x, get_logits=False):
        """
        predict the labels for dataset x
        :param get_logits: to return only logits
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
        if get_logits:
            return logits
        return tf.nn.sigmoid(logits, name='activationOutputLayer').eval()

    def test(self, x, y, old_model, history):
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            # Get old and new predictions
            old_output = old_model.predict_probabilities(x)
            y_old_correct = tf.cast(tf.equal(tf.round(old_output), y), tf.float32).eval()
            new_output = self.predict_probabilities(x)
            y_new_correct = tf.cast(tf.equal(tf.round(new_output), y), tf.float32).eval()
            likelihood = history.likelihood
            compatibility = tf.reduce_sum(y_old_correct * y_new_correct * likelihood) / tf.reduce_sum(y_old_correct * likelihood)
            return {'compatibility': compatibility.eval(), 'auc': sklearn.metrics.roc_auc_score(y, new_output)}


class History:
    """
    Class that implements the user's history, calculating means and vars.
    """
    def __init__(self, instances, labels, width_factor, epsilon):
        self.instances = instances
        self.labels = labels
        self.means = np.mean(instances, axis=0)
        self.vars = np.var(instances, axis=0) * width_factor + epsilon
        self.epsilon = epsilon
        self.width_factor = width_factor
        self.likelihood = None
        self.kernels = None

    def set_simple_likelihood(self, df):
        # compute likelihood for each attribute
        diff = np.subtract(df, self.means)
        sqr_diff = np.power(diff, 2)
        div = np.add(sqr_diff, self.vars)
        attribute_likelihoods = np.divide(self.vars, div)

        # todo: experimenting with likelihood here
        # merge the likelihood of all attributes
        self.likelihood = np.mean(attribute_likelihoods, axis=1)
        # self.likelihood = np.round(self.likelihood)
        # self.likelihood = 1 + self.likelihood
        self.likelihood = np.reshape(self.likelihood, (len(df), 1))

    def set_norm_dist_likelihood(self, df):
        # compute likelihood for each attribute
        sqr_diff = np.power(np.subtract(df, self.means), 2)
        power = np.divide(sqr_diff, 2 * self.vars)
        exp = np.exp(power)
        sqrt = np.sqrt(2*np.pi*self.vars)
        attribute_likelihoods = np.divide(1, sqrt * exp)

        # merge the likelihood of all attributes
        self.likelihood = np.product(attribute_likelihoods, axis=1)
        self.likelihood = np.reshape(self.likelihood, (len(df), 1))

    def set_cheat_likelihood(self, x):
        self.likelihood = []
        for index, row in x.iterrows():
            if row['ExternalRiskEstimate'] > 75:
                self.likelihood += [1]
            else:
                self.likelihood += [0]
        self.likelihood = np.reshape(self.likelihood, (len(x), 1))

    def set_non_parametric_likelihood(self, df, sigma):
        distances = []
        for instance in df:
            entry = []
            for hist_instance in self.instances:
                entry += [np.linalg.norm(instance - hist_instance)]
            distances += [entry]
        distances = np.asanyarray(distances)
        self.kernels = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*np.square(distances/sigma))

# ------------------ #
# todo: MODIFY THESE #
# ------------------ #

# Data-set paths

dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\creditRiskAssessment\\heloc_dataset_v1.csv'
results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\creditRiskAssessment.csv"
target_col = 'RiskPerformance'
dataset_fraction = 0.5

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\hospitalMortalityPrediction\\ADMISSIONS_encoded.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\hospitalMortalityPrediction.csv"
# target_col = 'HOSPITAL_EXPIRE_FLAG'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\recividismPrediction\\compas-scores-two-years_encoded.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\recividismPrediction.csv"
# target_col = 'is_recid'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\fraudDetection\\transactions.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\fraudDetection.csv"
# target_col = 'isFraud'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\e-learning_without_user_id.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\results\\e-learning.csv"
# target_col = 'correct'
# dataset_fraction = 0.03

df = pd.read_csv(dataset_path)

# creditRiskAssessment
users = {'1': df[:100].loc[df['ExternalRiskEstimate'] > 75]}.items()

Y = df.pop(target_col)
X = df.loc[:]

# extract dataset subset
rows = len(X)
X = X[:int(rows * dataset_fraction)]
Y = Y[:int(rows * dataset_fraction)]

# ---------------------------- #
# todo: MODIFY USER SIMULATION #
# ---------------------------- #

# history_test_x = X[100:].loc[df['ExternalRiskEstimate'] > 75]
# history_test_y = Y[100:].loc[df['ExternalRiskEstimate'] > 75]

# # e-learning:
#
# # # single user
# # user_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\e-learning_user_77912.csv'
# # user_instances = pd.read_csv(user_path)
# # user_instances.pop(target_col)
#
# # all users
# users_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\e-learning_with_user_id.csv'
# df = pd.read_csv(users_path)
# df = df[:int(len(df) * dataset_fraction)]
# del df['correct']
# users = df.groupby(['user_id'])

user_max_count = 1
user_count = 0

# min max scale and binarize the target labels
X_original = X
# history_test_x_original = history_test_x
scaler = MinMaxScaler()
X = scaler.fit_transform(X, Y)
label = LabelBinarizer()
Y = label.fit_transform(Y)

# history_test_x = scaler.fit_transform(history_test_x, history_test_y)
# history_test_y = label.fit_transform(history_test_y)

# compute base model:
# todo: h1 here

# h2_without_history_auc = {}
# new_correct = {}
got_without_history = False

# Data fractions
h1_train_fraction = 0.05
h2_train_fraction = 0.2

# Dissonance types
# diss_types = ["D", "D'", "D''"]
diss_types = ["D"]

# Dissonance weights in [0,100] as percentages
diss_weights = range(12)
factor = 30
repetitions = 1

for user_id, user_instances in users:
# for user_id, user_instances in users.items():
    if len(user_instances) < 5:
        continue
    user_count += 1
    if user_count > user_max_count:
        break
    try:
        del user_instances['user_id']
    except:
        pass
    print(str(user_count)+'/'+str(user_max_count)+' user id: '+str(user_id)+', instances: '+str(len(user_instances)))

    # get likelihood of dataset
    hist_labels = user_instances.pop(target_col)
    history = History(scaler.transform(user_instances), label.transform(hist_labels), 0.1, 0.0000001)
    # history.set_cheat_likelihood(X_original)
    history.set_simple_likelihood(X)
    # history.set_non_parametric_likelihood(X, 0.5)

    # history_test = History(scaler.transform(user_instances), 0.1, 0.0000001)
    # history_test.set_cheat_likelihood(history_test_x_original)
    # history_test.set_simple_likelihood(history_test_x)

    # train compatible models:
    # h2_without_history_x = []
    # h2_without_history_y = []
    # h2_with_history_x = []
    # h2_with_history_y = []

    iteration = 0

    # train_starts = range(21)
    train_starts = [2]
    h1_epochs_set = [1000]
    # h2_epochs_set = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    h2_epochs_set = [300]
    for train_start in train_starts:
        for h1_epochs in h1_epochs_set:
            for j in range(repetitions):
                h1 = MLP(X, Y, h1_train_fraction, h1_epochs, 50, 10, 5, 0.02,
                         train_start=train_start, initial_stdev=1, history=history, make_train_similar_to_history=True)

                # with open(results_path, 'w', newline='') as csv_file:
                #     writer = csv.writer(csv_file)
                #     row = ["compatibility", "h1 auc"]
                #     for diss_type in diss_types:
                #         row += ["h2 auc (" + diss_type + ") WITHOUT history"]
                #         row += ["h2 auc (" + diss_type + ") WITH history"]
                #     writer.writerow(row + ["dissonance weight", "user", "h1 train fraction = " + str(h1_train_fraction),
                #                            "h2 train fraction = " + str(h2_train_fraction)])

                print('train start = ' + str(train_start) + '/' + str(len(train_starts) - 1))

                # iteration = 0

                for h2_epochs in h2_epochs_set:

                    # todo: remove this
                    h2_without_history_x = []
                    h2_without_history_y = []
                    h2_with_history_x = []
                    h2_with_history_y = []
                    # h2_on_history_without_history_y = []
                    # h2_on_history_with_history_x = []
                    # h2_on_history_without_history_x = []
                    # h2_on_history_with_history_y = []

                    for i in diss_weights:

                        offset = 0
                        for use_history in [False, True]:
                            for diss_type in diss_types:

                                iteration += 1
                                iterations = len(diss_weights) * repetitions * len(diss_types) * 2
                                # if not got_without_history:
                                #     iterations *= 2
                                print("-------\n"+str(iteration)+"/"+str(iterations)+"\n-------")

                                # if not use_history and got_without_history:
                                #     # compute compatibility for this history
                                #     current_new_correct = new_correct[i]
                                #     numerator = old_correct * current_new_correct * likelihood
                                #     numerator_sum = np.sum(numerator)
                                #     denominator = old_correct * likelihood
                                #     denominator_sum = np.sum(denominator)
                                #     compatibility = numerator_sum / denominator_sum
                                #     h2_without_history_x += [compatibility]
                                #     h2_without_history_y += [h2_without_history_auc[i]]
                                #     print('test auc = '+str(h2_without_history_auc[i])+', compatibility = '+str(compatibility))
                                #     continue

                                diss_weight = factor * i / 100.0
                                # diss_weight = factor * i
                                tf.reset_default_graph()
                                h2 = MLP(X, Y, h2_train_fraction, h2_epochs, 50, 10, 5,
                                         0.02, diss_weight, h1, diss_type, True, True, history, use_history)

                                # result_on_history = h2.test(history_test_x, history_test_y, h1, history_test)
                                # result_on_history = h2.test(X, Y, h1, history)

                                # # save no history vectors for computing compatibility with different user histories
                                # if not use_history:
                                #     h2_without_history_auc[i] = h2.auc
                                #     new_correct[i] = h2.new_correct
                                #     old_correct = h2.old_correct
                                #     likelihood = history.likelihood[h2.test_indexes]

                                # with open(results_path, 'a', newline='') as csv_file:
                                #     writer = csv.writer(csv_file)
                                #     row = [str(h2.compatibility), str(h1.auc)]
                                #     for k in diss_types*2:
                                #         row += [""]
                                #     row += [str(diss_weight), str(user_id)]
                                #     row[2 + offset] = str(h2.auc)
                                #     writer.writerow(row)
                                # offset += 1

                                # for plot
                                if not use_history:
                                    h2_x = h2_without_history_x
                                    h2_y = h2_without_history_y
                                    # h2_on_history_x = h2_on_history_without_history_x
                                    # h2_on_history_y = h2_on_history_without_history_y
                                else:
                                    h2_x = h2_with_history_x
                                    h2_y = h2_with_history_y
                                    # h2_on_history_x = h2_on_history_with_history_x
                                    # h2_on_history_y = h2_on_history_with_history_y
                                h2_x += [h2.compatibility]
                                h2_y += [h2.auc]
                                # h2_on_history_x += [result_on_history['compatibility']]
                                # h2_on_history_y += [result_on_history['auc']]

                    # got_without_history = True

                    # plot
                    h1_x = [min(min(h2_without_history_x), min(h2_with_history_x)),
                            max(max(h2_without_history_x), max(h2_with_history_x))]
                    h1_y = [h1.auc, h1.auc]
                    plt.plot(h1_x, h1_y, 'k--', label='h1')
                    plt.plot(h2_without_history_x, h2_without_history_y, 'bs', label='h2 without history')
                    plt.plot(h2_with_history_x, h2_with_history_y, 'r^', label='h2 with history')
                    plt.xlabel('compatibility')
                    plt.ylabel('AUC')
                    plt.legend(('h1', 'h2 without history', 'h2 with history'), loc='center left')

                    # h1_x = [min(min(h2_without_history_x), min(h2_with_history_x),
                    #             min(h2_on_history_without_history_x), min(h2_on_history_with_history_x)),
                    #         max(max(h2_without_history_x), max(h2_with_history_x),
                    #             max(h2_on_history_without_history_x), max(h2_on_history_with_history_x))]
                    # h1_y = [h1.auc, h1.auc]
                    # plt.plot(h1_x, h1_y, 'k--', label='h1')
                    # plt.plot(h2_without_history_x, h2_without_history_y, 'bs', label='h2 without history')
                    # plt.plot(h2_with_history_x, h2_with_history_y, 'r^', label='h2 with history')
                    # plt.plot(h2_on_history_without_history_x, h2_on_history_without_history_y, 'gs', label='h2 on hist without history')
                    # plt.plot(h2_on_history_with_history_x, h2_on_history_with_history_y, 'y^', label='h2 on hist with history')
                    # plt.xlabel('compatibility')
                    # plt.ylabel('AUC')
                    # plt.legend(('h1', 'h2 without history', 'h2 with history',
                    #             'h2 on hist without history', 'h2 on hist with history'), loc='center left')

                    # plt.title('User ' + str(user_id))
                    plt.title('Train set ' + str(train_start)+', size='+str(len(h1.train_indexes)))
                    # plt.title('train '+str(train_start)+' h1 ' + str(h1_epochs) + ' h2 ' + str(h2_epochs))

                    # save plot
                    plots_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plots'
                    plot_count = len(os.listdir(plots_dir))
                    # plt.savefig(plots_dir + '\\' + str(plot_count + 1) + '_user_' + str(user_id) + '.png')
                    # plt.savefig(plots_dir + '\\' + str(plot_count + 1) + '_train_start_' + str(train_start) + '.png')
                    plt.savefig(plots_dir + '\\' + str(plot_count + 1) + '_h1_' + str(h1_epochs) + '_h2_' + str(h2_epochs) + '.png')

                    # plt.show()
                    plt.clf()
