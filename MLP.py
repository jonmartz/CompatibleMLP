import os.path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce


class MLP:
    """
    Class implementing a Multi Layered Perceptron, capable of training using the log loss + dissonance
    to be able to produce compatible updates.
    """

    def __init__(self, X, Y, train_fraction, train_epochs, batch_size, layer_1_neurons, layer_2_neurons, learning_rate,
                 diss_weight=None, old_model=None, dissonance_type=None, make_h1_subset=True, copy_h1_weights=True,
                 history=None, use_history=False, train_start=0, initial_stdev=1, make_train_similar_to_history=False,
                 model='L0', test_model=True):

        start_time = int(round(time.time() * 1000))

        # ------------------------------ #
        # PREPARE TRAINING AND TEST SETS #
        # ------------------------------ #

        # shuffle indexes to cover train and test sets
        if old_model is None or not make_h1_subset:
            # shuffled_indexes = np.random.randint(len(X), size=len(X))
            shuffled_indexes = range(len(X))
            # train_stop = int(len(X) * train_fraction)
            train_stop = train_fraction
            self.train_indexes = shuffled_indexes[:train_stop]
            self.test_indexes = shuffled_indexes[train_stop:]

            # # todo: trying to eliminate randomness here
            # indexes = range(len(X))
            # train_size = int(len(X) * train_fraction)
            # self.train_indexes = indexes[train_start * train_size:(train_start + 1) * train_size]

            # if make_train_similar_to_history:
            #     similar_indexes = []
            #     for index in self.train_indexes:
            #         # if history.likelihood[index] == 0:
            #         if history.likelihood[index] == 1:
            #             similar_indexes += [index]
            #     self.train_indexes = similar_indexes

            # self.test_indexes = [x for x in indexes if x not in self.train_indexes]

        else:  # make the old train set to be a subset of the new train set
            # shuffled = np.random.randint(len(old_model.test_indexes), size=len(old_model.test_indexes))
            # shuffled_test_indexes = old_model.test_indexes[shuffled]

            # todo: trying to eliminate randomness here
            shuffled_test_indexes = old_model.test_indexes
            # test_stop = int(len(X) * train_fraction - len(old_model.train_indexes))
            test_stop = train_fraction - len(old_model.train_indexes)
            self.train_indexes = np.concatenate((old_model.train_indexes, shuffled_test_indexes[:test_stop]))
            self.test_indexes = shuffled_test_indexes[test_stop:]

        # assign train and test subsets
        X_train = X[self.train_indexes]
        Y_train = Y[self.train_indexes]
        X_test = X[self.test_indexes]
        Y_test = Y[self.test_indexes]

        if history is not None:
            try:
                kernels_train = history.kernels[self.train_indexes]
                kernels_test = history.kernels[self.test_indexes]
            except TypeError:
                pass
            try:
                likelihood_train = history.likelihood[self.train_indexes]
                likelihood_test = history.likelihood[self.test_indexes]
            except TypeError:
                pass

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

        # set initial weights
        if old_model is None or not copy_h1_weights:
            # if True:
            w1_initial = tf.truncated_normal([n_features, layer_1_neurons], mean=0,
                                             stddev=initial_stdev / np.sqrt(n_features))
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

        # for non parametric compatibility
        if model in ['L1', 'L2']:
            kernels = tf.placeholder(tf.float32, [None, len(history.instances)], name='kernels')
            hist_x = tf.placeholder(tf.float32, [None, n_features], name='hist_input')
            hist_y = tf.placeholder(tf.float32, [None, labels_dim], name='hist_labels')
            hist_y_old_correct = tf.placeholder(tf.float32, [None, labels_dim], name='hist_old_corrects')

            hist_y1 = tf.sigmoid((tf.matmul(hist_x, W1) + b1), name='hist_activationLayer1')
            hist_y2 = tf.sigmoid((tf.matmul(hist_y1, W2) + b2), name='hist_activationLayer2')
            hist_logits = tf.matmul(hist_y2, Wo) + bo

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
                raise Exception("invalid dissonance type")

            if history is None:
                loss = log_loss + diss_weight * dissonance
                compatibility = tf.reduce_sum(y_old_correct * y_new_correct) / tf.reduce_sum(y_old_correct)
            else:
                if not use_history:
                    loss = log_loss + diss_weight * dissonance
                else:
                    if model in ['L1', 'L2']:

                        # product = kernels * hist_dissonance
                        # numerator = tf.reduce_sum(product, axis=1)
                        # denominator = tf.reduce_sum(kernels, axis=1)
                        # kernel_likelihood = numerator / denominator

                        if model == 'L1':
                            hist_dissonance = hist_y_old_correct * tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=hist_y,
                                logits=hist_logits,
                                name='hist_dissonance')
                            hist_dissonance = tf.reshape(hist_dissonance, [-1])
                            kernel_likelihood = tf.reduce_sum(kernels * hist_dissonance) / tf.reduce_sum(kernels)
                            loss = log_loss + diss_weight * kernel_likelihood
                        elif model == 'L2':
                            # shape = tf.shape(kernels)
                            kernel_likelihood = tf.reduce_sum(kernels, axis=1) / len(history.instances)
                            loss = log_loss + diss_weight * kernel_likelihood * dissonance
                    else:
                        loss = log_loss + diss_weight * dissonance * likelihood

                if model in ['L1', 'L2']:
                    # todo: maybe use average instead of sum?
                    compatibility = tf.reduce_sum(
                        y_old_correct * y_new_correct * tf.reduce_sum(kernels, axis=1)) / tf.reduce_sum(
                        y_old_correct * tf.reduce_sum(kernels, axis=1))
                else:
                    compatibility = tf.reduce_sum(y_old_correct * y_new_correct * likelihood) / tf.reduce_sum(
                        y_old_correct * likelihood)

        loss = loss / batch_size

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
                print("TRAINING h1: train size = " + str(train_fraction))
                # print("\nTRAINING h1:\ntrain fraction = " + str(int(100 * train_fraction)) + "%\n")

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

                if test_model:
                    acc, lss, out = sess.run([accuracy, loss, output], feed_dict={x: X_test, y: Y_test})
                    # self.auc = sklearn.metrics.roc_auc_score(Y_test, out)
                    self.accuracy = acc

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

                    # print("test auc = %.4f" % self.auc)
                    print("test acc = %.4f" % self.accuracy)

            else:  # with compatibility

                if history is not None:
                    if use_history:
                        history_string = "USE HISTORY"
                    else:
                        history_string = "IGNORE HISTORY"
                else:
                    history_string = "NO HISTORY"

                print("TRAINING h2: train size = " + str(train_fraction) + ", diss weight = " + str(diss_weight)
                      # str(int(100 * train_fraction)) + "%, diss weight = " + str(diss_weight)
                      + ", diss type = " + str(dissonance_type) + ", " + history_string)

                # get the old model predictions
                Y_train_old_probabilities = old_model.predict_probabilities(X_train)
                Y_train_old_labels = tf.round(Y_train_old_probabilities).eval()
                Y_train_old_correct = tf.cast(tf.equal(Y_train_old_labels, Y_train), tf.float32).eval()
                Y_test_old_probabilities = old_model.predict_probabilities(X_test)
                Y_test_old_labels = tf.round(Y_test_old_probabilities).eval()
                Y_test_old_correct = tf.cast(tf.equal(Y_test_old_labels, Y_test), tf.float32).eval()

                if model in ['L1', 'L2']:
                    hist_Y_old_probabilities = old_model.predict_probabilities(history.instances)
                    # hist_Y_old_probabilities = tf.nn.sigmoid(hist_Y_old_logits, name='hist_probabilities').eval()
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
                            sess.run(train_step,
                                     feed_dict={x: X_batch, y: Y_batch,
                                                y_old_probabilities: Y_batch_old_probabilities,
                                                y_old_correct: Y_batch_old_correct})
                        else:
                            if model in ['L1', 'L2']:
                                kernels_batch = kernels_train[batch_start:batch_end]
                                sess.run(train_step,
                                         feed_dict={x: X_batch, y: Y_batch,
                                                    y_old_probabilities: Y_batch_old_probabilities,
                                                    y_old_correct: Y_batch_old_correct,
                                                    hist_x: history.instances,
                                                    hist_y: history.labels,
                                                    hist_y_old_correct: hist_Y_old_correct,
                                                    kernels: kernels_batch})
                            else:
                                likelihood_batch = likelihood_train[batch_start:batch_end]
                                sess.run(train_step,
                                         feed_dict={x: X_batch, y: Y_batch,
                                                    y_old_probabilities: Y_batch_old_probabilities,
                                                    y_old_correct: Y_batch_old_correct,
                                                    likelihood: likelihood_batch})

                if test_model:
                    if history is None:
                        out, com, new_correct, acc = sess.run(
                            [output, compatibility, y_new_correct, accuracy],
                            feed_dict={x: X_test, y: Y_test,
                                       y_old_probabilities: Y_test_old_probabilities,
                                       y_old_correct: Y_test_old_correct})
                    else:
                        if model in ['L1', 'L2']:
                            if use_history:
                                out, com, new_correct, _hist_diss, _kernel_likelihood, acc = sess.run(
                                    [output, compatibility, y_new_correct, hist_dissonance, kernel_likelihood, accuracy],
                                    feed_dict={x: X_test, y: Y_test,
                                               y_old_probabilities: Y_test_old_probabilities,
                                               y_old_correct: Y_test_old_correct,
                                               hist_x: history.instances,
                                               hist_y: history.labels,
                                               hist_y_old_correct: hist_Y_old_correct,
                                               kernels: kernels_test})
                            else:
                                out, com, new_correct, acc = sess.run(
                                    [output, compatibility, y_new_correct, accuracy],
                                    feed_dict={x: X_test, y: Y_test,
                                               y_old_probabilities: Y_test_old_probabilities,
                                               y_old_correct: Y_test_old_correct,
                                               # hist_y: history.labels,
                                               # hist_y_old_correct: hist_Y_old_correct,
                                               kernels: kernels_test})
                        else:
                            out, com, new_correct, acc = sess.run(
                                [output, compatibility, y_new_correct, accuracy],
                                feed_dict={x: X_test, y: Y_test,
                                           y_old_probabilities: Y_test_old_probabilities,
                                           y_old_correct: Y_test_old_correct,
                                           likelihood: likelihood_test})

                    self.compatibility = com
                    # self.auc = sklearn.metrics.roc_auc_score(Y_test, out)
                    self.accuracy = acc
                    self.new_correct = new_correct
                    self.old_correct = Y_test_old_correct

                    # print("FINISHED:\ttest auc = %.4f" % self.auc + ", compatibility = %.4f" % self.compatibility)
                    print("test acc = %.4f" % self.accuracy + ", compatibility = %.4f" % self.compatibility)
                    # print("log loss = "+str(np.sum(log_lss)))
                    # print("dissonance = "+str(np.sum(diss)))

            # save weights
            self.final_W1 = W1.eval()
            self.final_b1 = b1.eval()
            self.final_W2 = W2.eval()
            self.final_b2 = b2.eval()
            self.final_Wo = Wo.eval()
            self.final_bo = bo.eval()

            runtime = str(int((round(time.time() * 1000)) - start_time) / 1000)
            print("runtime = " + str(runtime) + " secs\n")

    def predict_probabilities(self, x):
        """
        predict the labels for dataset x
        :param x: dataset to predict labels of
        :return: numpy array with the probability for each label
        """
        # _x = tf.cast(x, tf.float32).eval()
        # _y1 = tf.sigmoid((tf.matmul(_x, self.final_W1) + self.final_b1)).eval()
        # _y2 = tf.sigmoid((tf.matmul(_y1, self.final_W2) + self.final_b2)).eval()
        # _logits = (tf.matmul(_y2, self.final_Wo) + self.final_bo).eval()
        # _out = tf.nn.sigmoid(_logits, name='activationOutputLayer').eval()

        mul1 = np.matmul(x, self.final_W1) + self.final_b1
        y1 = 1 / (1 + np.exp(-mul1))
        mul2 = np.matmul(y1, self.final_W2) + self.final_b2
        y2 = 1 / (1 + np.exp(-mul2))
        logits = np.matmul(y2, self.final_Wo) + self.final_bo
        return 1 / (1 + np.exp(-logits))

    def test(self, x, y, old_model=None, history=None):
        # init_op = tf.global_variables_initializer()
        # with tf.Session() as sess:
        #     sess.run(init_op)
        #     new_output = self.predict_probabilities(x)
        #     correct_prediction = tf.equal(tf.round(new_output), y)
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy").eval()
        #
        #     if old_model is None:
        #         # return sklearn.metrics.roc_auc_score(y, new_output)
        #         return accuracy
        #
        #     old_output = old_model.predict_probabilities(x)
        #     y_old_correct = tf.cast(tf.equal(tf.round(old_output), y), tf.float32).eval()
        #     y_new_correct = tf.cast(tf.equal(tf.round(new_output), y), tf.float32).eval()
        #     if history is not None:
        #         likelihood = history.likelihood
        #         compatibility = tf.reduce_sum(y_old_correct * y_new_correct * likelihood) / tf.reduce_sum(
        #             y_old_correct * likelihood)
        #     else:
        #         compatibility = tf.reduce_sum(y_old_correct * y_new_correct) / tf.reduce_sum(y_old_correct)
        #     # return {'compatibility': compatibility.eval(), 'auc': sklearn.metrics.roc_auc_score(y, new_output)}
        #     return {'compatibility': compatibility.eval(), 'auc': accuracy}

        new_output = self.predict_probabilities(x)
        y_new_correct = np.equal(np.round(new_output), y).astype(int)
        accuracy = np.mean(y_new_correct)

        if old_model is None:
            # return sklearn.metrics.roc_auc_score(y, new_output)
            return accuracy

        old_output = old_model.predict_probabilities(x)
        y_old_correct = np.equal(np.round(old_output), y).astype(int)
        if history is not None:
            likelihood = history.likelihood
            compatibility = np.sum(y_old_correct * y_new_correct * likelihood) / np.sum(y_old_correct * likelihood)
        else:
            compatibility = np.sum(y_old_correct * y_new_correct) / np.sum(y_old_correct)
        # return {'compatibility': compatibility.eval(), 'auc': sklearn.metrics.roc_auc_score(y, new_output)}
        return {'compatibility': compatibility, 'auc': accuracy}


class History:
    """
    Class that implements the user's history, calculating means and vars.
    """

    def __init__(self, instances, labels=None, width_factor=0.1, epsilon=0.0000001):
        self.instances = instances
        self.labels = labels
        self.means = np.mean(instances, axis=0)
        self.vars = np.var(instances, axis=0) * width_factor + epsilon
        self.epsilon = epsilon
        self.width_factor = width_factor
        self.likelihood = None
        self.kernels = None

    def set_simple_likelihood(self, df, magnitude_multiplier=1):
        # compute likelihood for each attribute
        diff = np.subtract(df, self.means)
        sqr_diff = np.power(diff, 2)
        div = np.add(sqr_diff, self.vars)
        attribute_likelihoods = np.divide(self.vars, div) * magnitude_multiplier

        # todo: experimenting with likelihood here
        # merge the likelihood of all attributes
        self.likelihood = np.mean(attribute_likelihoods, axis=1)
        # self.likelihood = np.round(self.likelihood)
        # self.likelihood = 1 + self.likelihood
        self.likelihood = np.reshape(self.likelihood, (len(df), 1))

    def set_cheat_likelihood(self, df, threshold, likely_val=1):
        """
        Works only with credit risk data-set
        """
        self.likelihood = []
        for index, row in df.iterrows():
            if row['ExternalRiskEstimate'] > threshold:
                self.likelihood += [likely_val]
            else:
                self.likelihood += [0]
        self.likelihood = np.reshape(self.likelihood, (len(df), 1))

    def set_kernels(self, df, sigma=1, magnitude_multiplier=1):
        distances = []
        for instance in df:
            entry = []
            for hist_instance in self.instances:
                entry += [np.linalg.norm(instance - hist_instance)]
            distances += [entry]
        distances = np.asanyarray(distances)
        self.kernels = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(
            -1 / 2 * np.square(distances / sigma)) * magnitude_multiplier

# Data-set paths

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\creditRiskAssessment\\heloc_dataset_v1.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\creditRiskAssessment.csv"
# target_col = 'RiskPerformance'
# dataset_fraction = 0.5
# threshold = 75
# users = {'1': df[:100].loc[df['ExternalRiskEstimate'] > threshold]}.items()

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\hospitalMortalityPrediction\\ADMISSIONS_encoded.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\hospitalMortalityPrediction.csv"
# target_col = 'HOSPITAL_EXPIRE_FLAG'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\recividismPrediction\\compas-scores-two-years_encoded.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\recividismPrediction.csv"
# target_col = 'is_recid'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\fraudDetection\\transactions.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\fraudDetection.csv"
# target_col = 'isFraud'

full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\e-learning_full_encoded.csv'
# balanced_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\e-learning_balanced_encoded.csv'
results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\results\\e-learning.csv"
target_col = 'correct'
categ_cols = ['tutor_mode', 'answer_type', 'type']
# user_group_names = ['user_id', 'teacher_id', 'student_class_id']
user_group_names = ['user_id']
dataset_fraction = 0.4
df_train_fraction = 0.1
h1_train_fraction = 200
h2_train_fraction = 5000
h1_epochs = 1000
h2_epochs = 100
diss_weights = range(6)
diss_multiply_factor = 50
repetitions = [0]
# repetitions = range(10)
random_state = 1
min_history_size = 200
max_history_size = 3000

# full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\KddCup\\2006\\kddCup_full_encoded.csv'
# balanced_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\KddCup\\2006\\kddCup_balanced_encoded.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\results\\kddCup.csv"
# target_col = 'Correct First Attempt'
# # categ_cols = ['Problem Hierarchy']
# categ_cols = []
# user_group_names = ['Student']
# dataset_fraction = 1.0
# h1_train_fraction = 200
# h2_train_fraction = 3000
# diss_weights = range(10)
# diss_multiply_factor = 30

# pre-process data

df_full = pd.read_csv(full_dataset_path)
# df_balanced = pd.read_csv(balanced_dataset_path)
# df = df[:int(len(df.index) * dataset_fraction)]

try:
    del df_full['school_id']
    del df_full['teacher_id']
    del df_full['student_class_id']
except:
    pass

# one hot encoding
ohe = ce.OneHotEncoder(cols=categ_cols, use_cat_names=True)
df_full = ohe.fit_transform(df_full)
# df_balanced = ohe.transform(df_balanced)

print('pre-processing data and splitting into train and test sets...')

# create user groups
user_groups_train = []
user_groups_test = []
for user_group_name in user_group_names:
    user_groups_test += [df_full.groupby([user_group_name])]

# separate histories into training and test sets
# todo: make generic and not for only first group
students_group = user_groups_test[0]
df_train = students_group.apply(lambda x: x.sample(n=int(len(x)*df_train_fraction)+1, random_state=random_state))
df_train.index = df_train.index.droplevel(0)
user_groups_test[0] = df_full.drop(df_train.index).groupby([user_group_names[0]])
user_groups_train += [df_train.groupby([user_group_names[0]])]

# balance train set
del df_train[user_group_names[0]]
target_group = df_train.groupby(target_col)
df_train = target_group.apply(lambda x: x.sample(target_group.size().min(), random_state=random_state))
df_train = df_train.reset_index(drop=True)

first_repetition = True
for repetition in repetitions:

    df_train_subset = df_train.sample(frac=dataset_fraction, random_state=random_state)

    test_group = {str(repetition + 1): df_train_subset}.items()
    if first_repetition:
        first_repetition = False
        user_group_names.insert(0, 'test')
        user_groups_test.insert(0, test_group)
    else:
        user_groups_test[0] = test_group

    X = df_train_subset.loc[:, df_train_subset.columns != target_col]
    Y = df_train_subset[[target_col]]

    X = X[:5000]
    Y = Y[:5000]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X, Y)
    labelizer = LabelBinarizer()
    Y = labelizer.fit_transform(Y)

    # history_test_x = X[100:].loc[df['ExternalRiskEstimate'] > 75]
    # history_test_y = Y[100:].loc[df['ExternalRiskEstimate'] > 75]

    # min max scale and binarize the target labels
    # X_original = X
    # history_test_x_original = history_test_x

    # history_test_x = scaler.fit_transform(history_test_x, history_test_y)
    # history_test_y = label.fit_transform(history_test_y)

    # h2_without_history_auc = {}
    # new_correct = {}
    # got_without_history = False

    # Dissonance types
    # diss_types = ["D", "D'", "D''"]
    # diss_types = ["D"]

    # user_max_count = 10

    # h1s = []

    # com_range = []
    # acc_range = []
    # h1_train_fractions = range(200, h2_train_fraction, 200)

    # h1_train_fractions = [200]
    # for h1_train_fraction in h1_train_fractions:
    h1 = MLP(X, Y, h1_train_fraction, h1_epochs, 50, 10, 5, 0.02)

    print("training h2s not using history...")

    h2s_not_using_history = []
    for i in diss_weights:
        print(str(len(h2s_not_using_history) + 1) + "/" + str(len(diss_weights)))
        diss_weight = diss_multiply_factor * i / 100.0
        h2s_not_using_history += [MLP(X, Y, h2_train_fraction, h2_epochs, 50, 10, 5, 0.02, diss_weight, h1, 'D', True, True, test_model=False)]
        tf.reset_default_graph()

    # com_range += [abs(h2s[0].compatibility-h2s[1].compatibility)]
    # acc_range += [abs(h2s[0].accuracy-h2s[1].accuracy)]
    # h1_train_fractions = [x / h2_train_fraction for x in h1_train_fractions]
    # plt.plot(h1_train_fractions, com_range, 'b', label='coms', marker='.')
    # plt.plot(h1_train_fractions, acc_range, 'r', label='accs', marker='.')
    # plt.xlabel('h1 train size / h2 train size')
    # # plt.ylabel('')
    # plt.legend(('compatibility range', 'accuracy range'), loc='upper right')
    # plt.title('Varying h1 train sizes, where h2 train size = '+str(h2_train_fraction))
    # plt.savefig('C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plots\\different_h1_train_sizes.png')
    # plt.show()
    # break

    user_group_idx = -1
    for user_group_test in user_groups_test:
        user_group_idx += 1
        user_group_name = user_group_names[user_group_idx]

        # if user_group_name == 'user_id' or user_group_name == 'teacher_id' or user_group_name == 'student_class_id':
        # if user_group_name == 'teacher_id' or user_group_name == 'student_class_id':
        #     continue

        directory = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plots\\' + user_group_name
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs(directory + '\\by_hist_length')
            os.makedirs(directory + '\\by_accuracy_range')
            os.makedirs(directory + '\\by_compatibility_range')

        total_users = 0
        for user_id, user_test_set in user_group_test:
            if len(user_test_set) >= min_history_size:
                total_users += 1

        user_count = 0
        for user_id, user_test_set in user_group_test:
            if user_group_name != 'test':
                if len(user_test_set) < min_history_size:
                    continue
                if len(user_test_set) > max_history_size:
                    continue
                # if user_count > user_max_count:
                #     continue
            user_count += 1

            for name in user_group_names:
                try:
                    del user_test_set[name]
                except:
                    pass
            print(
                str(user_count) + '/' + str(total_users) + ' ' + user_group_name + ' ' + str(user_id) + ', instances: ' + str(
                    len(user_test_set)) + '\n')

            history_test_x = scaler.transform(user_test_set.loc[:, user_test_set.columns != target_col])
            history_test_y = labelizer.transform(user_test_set[[target_col]])

            h1_acc = h1.test(history_test_x, history_test_y)

            h2_on_history_not_using_history_x = []
            h2_on_history_not_using_history_y = []
            for h2 in h2s_not_using_history:
                result_not_using_history = h2.test(history_test_x, history_test_y, h1)
                h2_on_history_not_using_history_x += [result_not_using_history['compatibility']]
                h2_on_history_not_using_history_y += [result_not_using_history['auc']]

            min_x = min(h2_on_history_not_using_history_x)
            max_x = max(h2_on_history_not_using_history_x)
            min_y = min(h2_on_history_not_using_history_y)
            max_y = max(h2_on_history_not_using_history_y)

            if user_group_name != 'test':

                user_train_set = user_groups_train[user_group_idx - 1].get_group(user_id)
                for name in user_group_names:
                    try:
                        del user_test_set[name]
                    except:
                        pass
                history_train_x = scaler.transform(user_train_set.loc[:, user_train_set.columns != target_col])
                history_train_y = labelizer.transform(user_train_set[[target_col]])

                # history = History(history_test_x, history_test_y, 0.01)
                history = History(history_train_x, history_train_y, 0.01)
                history.set_simple_likelihood(X, 2)
                # history.set_cheat_likelihood(X_original, threshold)
                history.set_kernels(X, magnitude_multiplier=10)

                models = ['L0', 'L1', 'L2']
                h2_on_history_L0_x = []
                h2_on_history_L0_y = []
                h2_on_history_L1_x = []
                h2_on_history_L1_y = []
                h2_on_history_L2_x = []
                h2_on_history_L2_y = []
                h2_on_history_x = [h2_on_history_L0_x, h2_on_history_L1_x, h2_on_history_L2_x]
                h2_on_history_y = [h2_on_history_L0_y, h2_on_history_L1_y, h2_on_history_L2_y]

                iteration = 0
                for i in diss_weights:
                    iteration += 1
                    print(str(iteration) + "/" + str(len(diss_weights)))
                    diss_weight = diss_multiply_factor * i / 100.0
                    for j in range(len(models)):
                        tf.reset_default_graph()
                        h2_using_history = MLP(X, Y, h2_train_fraction, h2_epochs, 50, 10, 5, 0.02, diss_weight, h1, 'D',
                                               history=history, use_history=True, model=models[j], test_model=False)
                        result_using_history = h2_using_history.test(history_test_x, history_test_y, h1)
                        h2_on_history_x[j] += [result_using_history['compatibility']]
                        h2_on_history_y[j] += [result_using_history['auc']]

                # PLOT
                # all hist approaches and no hist plot
                for model in h2_on_history_x:
                    min_model = min(model)
                    if min_x > min_model:
                        min_x = min_model
                    max_model = max(model)
                    if max_x < max_model:
                        max_x = max_model
                for model in h2_on_history_y:
                    min_model = min(model)
                    if min_y > min_model:
                        min_y = min_model
                    max_model = max(model)
                    if max_y < max_model:
                        max_y = max_model
                        
                h1_x = [min_x, max_x]
                h1_y = [h1_acc, h1_acc]
                plt.plot(h1_x, h1_y, 'k--', label='h1')
                plt.plot(h2_on_history_not_using_history_x, h2_on_history_not_using_history_y, 'b', marker='o', linewidth=2, label='h2 without history')
                plt.plot(h2_on_history_L0_x, h2_on_history_L0_y, 'r', marker='.', label='h2 with L0')
                plt.plot(h2_on_history_L1_x, h2_on_history_L1_y, 'm', marker='.', label='h2 with L1')
                plt.plot(h2_on_history_L2_x, h2_on_history_L2_y, 'orange', marker='.', label='h2 with L2')
                plt.xlabel('compatibility')
                plt.ylabel('accuracy')
                plt.legend(('h1', 'h2 not using history', 'h2 using L0', 'h2 using L1', 'h2 using L2'), loc='center left')

            else:  # on test
                h1_x = [min_x, max_x]
                h1_y = [h1_acc, h1_acc]
                plt.plot(h1_x, h1_y, 'k--', label='h1')
                plt.plot(h2_on_history_not_using_history_x, h2_on_history_not_using_history_y, 'b', marker='.', label='h2')
                plt.xlabel('compatibility')
                plt.ylabel('accuracy')
                plt.legend(('h1', 'h2'), loc='center left')
            # # hist and no hist plot
            # h1_x = [min(min(h2_on_history_not_using_history_x), min(h2_on_history_L0_x)),
            #         max(max(h2_on_history_not_using_history_x), max(h2_on_history_L0_x))]
            # h1_y = [h1.accuracy, h1.accuracy]
            # plt.plot(h1_x, h1_y, 'k--', label='h1')
            # plt.plot(h2_on_history_not_using_history_x, h2_on_history_not_using_history_y, 'b', marker='o', linewidth=2,
            #          label='h2 without history')
            # plt.plot(h2_on_history_L0_x, h2_on_history_L0_y, 'r', marker='.', label='h2 with history')
            # plt.xlabel('compatibility')
            # plt.ylabel('accuracy')
            # plt.legend(('h1', 'h2 not using history', 'h2 using history'), loc='center left')

            # # hist and no hist, and on hist test set plot
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

            # plt.title('Train set ' + str(train_start)+', size='+str(len(h1.train_indexes)))

            # save plot
            # plots_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plots'
            plots_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plots\\' + user_group_name
            plot_count = len(os.listdir(plots_dir))

            com_range = int(100 * (max_x - min_x))
            auc_range = int(100 * (max_y - min_y))

            if user_group_name == 'test':
                plt.title(user_group_name + ' ' + str(repetition + 1) + ', train sets: h1=' + str(h1_train_fraction) + ' h2='+ str(h2_train_fraction))
                plt.savefig(plots_dir + '\\test_'+str(repetition + 1)+'.png')
                plt.show()
            else:
                plt.title(user_group_name + ' ' + str(user_id) + ', ' + str(len(history_test_y)) + ' instances')
                plt.savefig(plots_dir + '\\by_hist_length\\len_' + str(len(user_test_set)) + '_' + user_group_name + '_' + str(user_id) + '.png')
                plt.savefig(plots_dir + '\\by_accuracy_range\\acc_' + str(auc_range) +'_' + user_group_name + '_' + str(user_id) + '.png')
                plt.savefig(plots_dir + '\\by_compatibility_range\\com_' + str(com_range) +'_' + user_group_name + '_' + str(user_id) + '.png')
            plt.clf()
