# A Binary Multilayerperceptron Classifier. Currently Depends on a custom
# dataset class defined in data_frame.py.
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import time

from sklearn.metrics import roc_auc_score, roc_curve
from .draw_nn import PlotNN

class BinaryMLP:
    """A Binary Classifier using a Multilayerperceptron.

    Makes probability predictions on a set of features (A 1-dimensional
    numpy vector belonging either to the 'signal' or the 'background'.

    Arguments:
    ----------------
    n_variables (int) :
    The number of input features.
    h_layers (list):
    A list representing the hidden layers. Each entry gives the number of
    neurons in the equivalent layer.
    savedir (str) :
    Path to the directory the model should be saved in.
    activation (str) :
    Default is 'relu'. Activation function used in the model. Also possible 
    is 'tanh' or 'sigmoid'.
    var_names (str) :
    Optional. If given this string is plotted in the controll plots title.
    Should describe the used dataset. 
    """

    def __init__(self, n_variables, h_layers, savedir, activation='relu',
                 var_names=None):
        self.n_variables = n_variables
        self.n_labels = 1
        self.h_layers = h_layers
        self.activation = activation
        self.name = savedir.rsplit('/')[-1]
        self.variables = var_names
        self.savedir = savedir

        # check wether model file exists
        if os.path.exists(self.savedir + '/{}.ckpt.meta'.format(self.name)):
            self.trained = True
        else:
            self.trained = False

        # create save directory if needed
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)

    def _get_parameters(self):
        """Create TensorFlow variables in two lists.
        
        Weights are initialized as random number taken from a normal
        distribution with standard deviation sqrt(2.0/n_inputs), where n_inputs
        is the number of neurons in the previous layer. Biases are initialized
        as 0.1.
        
        Returns:
        --------------
        weights (list) :
        A dictionary with the tensorflow Variables for the weights.
        biases (list) :
        A dictionary with the tensorflow Variables for the biases.
        """
        n_variables = self.n_variables
        h_layers = self.h_layers

        weights = [tf.Variable(
            tf.random_normal(shape = [n_variables, h_layers[0]],
                             stddev = tf.sqrt(2.0/n_variables)), name = 'W_1')]
        biases = [tf.Variable(tf.fill(dims=[h_layers[0]],
                                      value=0.1), name = 'B_1')]

        # fancy loop in order to create weights connecting the hidden layer.
        if len(h_layers) > 1:
            for i in range(1, len(h_layers)):
                weights.append(tf.Variable(
                    tf.random_normal( shape = [h_layers[i-1], h_layers[i]],
                    stddev = tf.sqrt(2.0/h_layers[i-1])),
                    name = 'W_{}'.format(i+1)))
                biases.append(tf.Variable(
                    tf.fill(dims=[h_layers[i]], value=0.1),
                    name = 'B_{}'.format(i+1)))

        weights.append(tf.Variable(
                tf.random_normal([h_layers[-1], 1],
                                 stddev=tf.sqrt(2.0/h_layers[-1])),
            name = 'W_out'))
        biases.append(tf.Variable(tf.fill(dims=[1], value=0.1),
                      name = 'B_out'))

        return weights, biases
    
    def _get_activation(self, activation):
        """Get activation function.

        Arguments:
        -----------
        activation (str) :
        Activation function which should be use. Has to be 'relu', 'tanh' or
        'sigmoid'.

        Returns:
        -----------
        activation (tf.nn.activation) :
        Chosen activation function.
        """
        
        if activation == 'tanh':
            return tf.nn.tanh
        elif activation =='sigmoid':
            return tf.nn.sigmoid
        elif activation == 'relu':
            return tf.nn.relu
        else:
            sys.exit('Choose activation function from "relu", ' +
                     '"tanh" or "sigmoid"')
            
    def _get_optimizer(self):
        """Get Opitimizer to be used for minimizing the loss function.

        Returns:
        ------------
        opt (tf.train.Optimizer) :
        Chosen optimizer.
        global_step (tf.Variable) :
        Variable that counts the minimization steps. Not trainable, only used 
        for internal bookkeeping.
        """
        
        global_step = tf.Variable(0, trainable=False)
        if self._lr_decay:
            learning_rate = (tf.train
                             .exponential_decay(self._lr, global_step,
                                                decay_steps = self._lr_decay[1],
                                                 decay_rate = self._lr_decay[0])
                             )
        else:
            learning_rate = self._lr

        if self._optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate)
        elif self._optimizer =='gradientdescent':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif self._optimizer == 'momentum':
            if self._momentum:
                opt = tf.train.MomentumOptimizer(learning_rate,
                                                 self._momentum[0],
                                                 use_nesterov = self._momentum[1])
            else:
                sys.exit('No momentum term for "momentum" optimizer available.')
        else :
            sys.exit('Choose Optimizer: "adam", ' +
            '"gradientdescent" or "momentum"')

        return opt, global_step
    
    def _model(self, x, W, B, keep_prob=1.0):
        """Model for the multi layer perceptron

        Arguments:
        --------------
        data (tf.placeholder) : 
        A tensorflow placeholder.
        W (list) :
        A list with the tensorflow Variables for the weights.
        B (list) :
        A list with the tensorflow Variables for the biases.

        Returns:
        ---------------
        output_node (tf.tensor)
        Prediction of the model.
        """
        activation = self._get_activation(self.activation)
        
        layer = tf.nn.dropout(activation(
            tf.add(tf.matmul(x, W[0]), B[0])), keep_prob)
        
        # fancy loop for creating hidden layer
        if len(self.h_layers) > 1:
            for weight, bias in zip(W[1:-1], B[1:-1]):
                layer = tf.nn.dropout(activation(
                    tf.add(tf.matmul(layer, weight), bias)),keep_prob)

        logit = tf.add(tf.matmul(layer, W[-1]), B[-1]) 

        return logit

    def train(self, train_data, val_data, epochs=10, batch_size=128,
              lr=1e-3, optimizer='adam', momentum=None, lr_decay=None,
              early_stop=10,  keep_prob=1.0, beta=0.0):
        """Train Neural Network with given training data set.

        Arguments:
        -------------
        train_data (custom dataset) :
        Contains training data.
        val_data (custom dataset) :
        Contains validation data.
        savedir (string) :
        Path to directory to save Plots.
        epochs (int) :
        Number of iterations over the whole trainig set.
        batch_size (int) :
        Number of batches fed into on optimization step.
        lr (float) :
        Default is 1e-3. Learning rate use by the optimizer for minizing the
        loss. The default value should work with 'adam'. Other optimizers may
        require other values. Tweak the learning rate for best results.
        optimizer (str) :
        Default is 'Adam'. Other options are 'gradientdescent' or 'momentum'.
        momentum list(float, bool) :
        Default is [0.1, True]. Only used if 'momentum' is used as optimizer.
        momentum[0] is the momentum, momentum[1] indicates, wether Nesterov
        momentum should be used.
        lr_decay list() :
        Default is None. Only use if 'gradientdescent' or 'momentum' is used as
        optimizer. List requires form [decay_rate (float), decay_steps (int)].
        List parameters are used to for exponentailly decaying learning rate.
        Try [0.96, 100000].
        early_stop (int):
        Default is 20. If Validation AUC has not increase  in the given number
        of epochs, the training is stopped. Only the model with the highest 
        validation auc score is saved.
        keep_prob (float):
        Probability of a neuron to 'activate'.
        beta (float):
        L2 regularization coefficient. Defaul 0.0 = regularization off.
        """
        self._lr = lr
        self._optimizer = optimizer
        self._momentum = momentum
        self._lr_decay = lr_decay
        
        train_graph = tf.Graph()
        with train_graph.as_default():
            x = tf.placeholder(tf.float32, [None, self.n_variables])
            y = tf.placeholder(tf.float32, [None, 1])
            w = tf.placeholder(tf.float32, [None, 1])

            x_mean = tf.Variable(np.mean(train_data.x, axis=0).astype(np.float32),
                                 trainable=False,  name='x_mean')
            x_std = tf.Variable(np.std(train_data.x, axis=0).astype(np.float32),
                                trainable=False,  name='x_std')
            x_scaled = tf.div(tf.sub(x, x_mean), x_std, name='x_scaled')

            weights, biases = self._get_parameters()

            # prediction, y_ is used for training, yy_ used for makin new
            # predictions
            y_ = self._model(x_scaled, weights, biases, keep_prob)
            yy_ = tf.nn.sigmoid(self._model(x_scaled, weights, biases))

            # loss function
            xentropy = tf.nn.sigmoid_cross_entropy_with_logits(y_, y)
            l2_reg = beta*self._l2_regularization(weights)
            loss = tf.add(tf.reduce_mean(tf.mul(w, xentropy)), l2_reg,
                                  name='loss')

            # optimizer
            opt, global_step = self._get_optimizer()
            train_step = opt.minimize(loss, global_step=global_step)

            # initialize the variables
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(weights + biases + [x_mean, x_std])

        

        # dont allocate all available gpu memory, remove if you can dont share a
        # machine with others
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.1
        with tf.Session(config=config, graph=train_graph) as sess:
            self.model_loc = self.savedir + '/{}.ckpt'.format(self.name)
            sess.run(init)
            
            train_auc = []
            val_auc = []
            train_loss = []
            early_stopping  = {'auc': 0.0, 'epoch': 0}
            epoch_durations = []
            print(90*'-')
            print('Train model: {}'.format(self.model_loc))
            print(90*'-')
            print('{:^20} | {:^20} | {:^20} |{:^25}'.format(
                'Epoch', 'Training Loss','AUC Training Score',
                'AUC Validation Score'))
            print(90*'-')

            for epoch in range(1, epochs+1):
                epoch_start = time.time()
                total_batches = int(train_data.n/batch_size)
                epoch_loss = 0
                for _ in range(total_batches):
                    train_x, train_y, train_w= train_data.next_batch(batch_size) 
                    batch_loss, _  = sess.run([loss, train_step],
                                              {x: train_x,
                                               y: train_y,
                                               w: train_w})
                    epoch_loss += batch_loss
                    
                # monitor training
                train_data.shuffle()
                train_loss.append(epoch_loss/total_batches)
                train_pre = []
                for batch in range(0, total_batches):
                    train_x, _, _ = train_data.next_batch(batch_size)
                    pred = sess.run(yy_, {x: train_x})
                    train_pre.append(pred)
                train_pre = np.concatenate(train_pre, axis=0)
                train_auc.append(
                    roc_auc_score(train_data.y[:total_batches*batch_size],
                                  train_pre))
                val_pre = sess.run(yy_, {x : val_data.x})
                val_auc.append(roc_auc_score(val_data.y, val_pre))
                
                print('{:^20} | {:^20.4e} | {:^20.4f} | {:^25.4f}'
                      .format(epoch, train_loss[-1],
                              train_auc[-1], val_auc[-1]))

                if early_stop:
                    # check for early stopping, only save model if val_auc has
                    # increased
                    if val_auc[-1] > early_stopping['auc']:
                        save_path = saver.save(sess, self.model_loc)
                        early_stopping['auc'] = val_auc[-1]
                        early_stopping['epoch'] = epoch
                        early_stopping['val_pre'] = val_pre
                    elif (epoch - early_stopping['epoch']) >= early_stop:
                        print(125*'-')
                        print('Validation AUC has not increased for {} epochs. ' \
                              'Achieved best validation auc score of {:.4f} ' \
                              'in epoch {}'.format(early_stop, 
                                  early_stopping['auc'],
                                  early_stopping['epoch']))
                        break
                else:
                    save_path = saver.save(sess, self.model_loc)
                
                # set internal dataframe index to 0
                
                epoch_end = time.time()
                epoch_durations.append(epoch_end - epoch_start)

            # get training prediction for validation, use batches to prevent
            # allocating to much gpu memory
            train_data.shuffle()
            evts_per_batch = 100
            total_batches = int(train_data.n/evts_per_batch)
            train_pre = []
            for batch in range(0, total_batches):
                train_x, _, _ = train_data.next_batch(evts_per_batch)
                pred = sess.run(yy_, {x: train_x})
                train_pre.append(pred)
            train_pre = np.concatenate(train_pre, axis=0)
            print(90*'-')
            self._validation(train_pre, train_data.y[:total_batches*evts_per_batch],
                             early_stopping['val_pre'], val_data.y)
            self._plot_auc_dev(train_auc, val_auc, early_stopping['epoch'])
            self._plot_loss(train_loss)
            self.trained = True
            self._write_parameters(batch_size, keep_prob, beta,
                                   np.mean(epoch_durations), early_stopping)
            print('Model saved in: {}'.format(save_path))
            print(90*'-')
            
    def _l2_regularization(self, weights):
        """Calculate and adds the squared values of the weights. This is used
        for L2 Regularization.
        """
        weights = map(lambda x: tf.nn.l2_loss(x), weights)

        return tf.add_n(weights)

    def classify(self, data):
        """Predict probability of a new feauture to belong to the signal.

        Arguments:
        ----------------
        data (custom data set):
        Data to classify.

        Returns:
        ----------------
        prob (np.array):
        Contains probabilities of a sample to belong to the signal.
        """
        if not self.trained:
            sys.exit('Model {} has not been trained yet'.format(self.name))

        predict_graph = tf.Graph()
        with predict_graph.as_default():
            weights, biases = self._get_parameters()
            x = tf.placeholder(tf.float32, [None, self.n_variables])
            x_mean = tf.Variable(-1.0, validate_shape=False,  name='x_mean')
            x_std = tf.Variable(-1.0, validate_shape=False,  name='x_std')
            x_scaled = tf.div(tf.sub(x, x_mean), x_std, name='x_scaled')
            y=  tf.nn.sigmoid(self._model(x_scaled, weights, biases))
            
            saver = tf.train.Saver()

        # dont allocate all available gpu memory, remove if you can dont share a
        # machine with others
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config, graph = predict_graph) as sess:
            saver.restore(sess, self.savedir + '/{}.ckpt'.format(self.name))
            prob = sess.run(y, {x: data})

        return prob

    def export_graph_to_pb(self):
        """Export the graph definition and the parameters of the model to a 
        Protobuff file, that can be used by TensorFlow's C++ API. Do not change
        node names.
        """
        if not self.trained:
            sys.exit('Model {} has not been trained yet'.format(self.name))

        export_graph = tf.Graph()
        with export_graph.as_default():
            weights, biases = self._get_parameters()
            x = tf.constant(-1.0, shape=[1, self.n_variables],
                               name='input_node')
            x_mean = tf.Variable(-1.0, validate_shape=False,  name='x_mean')
            x_std = tf.Variable(-1.0, validate_shape=False,  name='x_std')
            x_scaled = tf.div(tf.sub(x, x_mean), x_std, name='x_scaled')
            y = tf.nn.sigmoid(self._model(x_scaled, weights, biases),
                             name='output_node')
            
            saver = tf.train.Saver()

        # dont allocate all available gpu memory, remove if you can dont share a
        # machine with others
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config, graph=export_graph) as sess:
            saver.restore(sess, self.savedir + '/{}.ckpt'.format(self.name))
            const_graph=tf.graph_util.convert_variables_to_constants(
                sess, export_graph.as_graph_def(), ['output_node'])
            tf.train.write_graph(const_graph, self.savedir, self.name + ".pb",
                                 as_text=False)

    def _write_parameters(self, batch_size, keep_prob, beta, time,
                          early_stop):
        """Writes network parameters in a .txt file
        """

        with open('{}/NN_Info.txt'.format(self.savedir), 'w') as f:
            f.write('Number of input variables: {}\n'.format(self.n_variables))
            f.write('Number of hidden layers and neurons: {}\n'
                    .format(self.h_layers))
            f.write('Activation function: {}\n'.format(self.activation))
            f.write('Optimizer: {}, Learning Rate: {}\n'
                    .format(self._optimizer, self._lr))
            if self._momentum:
                f.write('Momentum: {}, Nesterov: {}\n'
                        .format(self._momentum[0], self._momentum[1]))
            if self._lr_decay:
                f.write('Decay rate: {}, Decay steps: {}\n'
                        .format(self._lr_decay[0], self._lr_decay[1]))
            f.write('Number of epochs trained: {}\n'
                    .format(early_stop['epoch']))
            f.write('Validation ROC-AUC score: {:.4f}\n'
                    .format(early_stop['auc']))
            f.write('Batch Size: {}\n'.format(batch_size))
            f.write('Dropout: {}\n'.format(keep_prob))
            f.write('L2 Regularization: {}\n'.format(beta))
            f.write('Mean Training Time per Epoch: {} s\n'.format(time))

        with open('{}/NN_Info.txt'.format(self.savedir), 'r') as f:
            for line in f:
                print(line)
        print(90*'-')

    def _validation(self, t_pred, t_labels, v_pred, v_labels):
        """Validation of the training process.
        Makes plots of ROC curves and displays the development of AUC score.

        Arguments:
        ----------------
        pred (np.array, shape(-1,)) :
        Predictions for data put in the model.
        labels (np.array, shape(-1)) :
        Lables of the validation dataset.
        """
        
        def seperate_sig_bkg(pred, labels):
            """This functions seperates signal and background output of the
            neural network.
            """
            y = np.hstack((pred, labels))
            sig = y[y[:,1]==1, 0]
            bg = y[y[:,1]==0, 0]
            return sig, bg
        
        # plot distribution
        t_sig, t_bg = seperate_sig_bkg(t_pred, t_labels)
        v_sig, v_bg = seperate_sig_bkg(v_pred, v_labels)
        bin_edges = np.linspace(0, 1, 30)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        v_sig = np.histogram(v_sig, bins=bin_edges, normed=True)[0]
        v_bg = np.histogram(v_bg, bins=bin_edges, normed=True)[0]
        
        plt.hist(t_sig, bins=bin_edges, histtype='step', lw=1.5,
                 label='Signal (Training)', normed='True', color='#1f77b4')
        plt.hist(t_bg, bins=bin_edges, lw=1.5, histtype='step',
                 label='Untergrund (Training)', normed='True',
                 color='#d62728')
        plt.plot(bin_centres, v_sig, ls='', marker='o', markersize=3, color='#1f77b4',
                 label='Signal (Validierung)')
        plt.plot(bin_centres, v_bg, ls='', marker='o', markersize=3, color='#d62728',
                 label='Untergrund (Validierung)')
        plt.ylim([0.0, np.amax(v_sig)*1.4])
        
        plt.legend(loc='upper left')
        plt.xlabel('Netzwerk Ausgabe')
        plt.ylabel('Ereignisse (normiert)')
        if self.variables:
            plt.title(self.variables, loc='left')
        plt.title(self.name, loc='center')
        # plt.title('CMS Private Work', loc='right')
        
        
        plt_name = self.name + '_dist'
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()

        # roc curve
        v_pred = np.reshape(v_pred, -1)
        fpr, tpr, thresh = roc_curve(v_labels, v_pred)
        auc = roc_auc_score(v_labels, v_pred)
        #plot the roc_curve
        plt_name = self.name +  '_roc'
        
        plt.plot(tpr, np.ones(len(fpr)) - fpr, color='#1f77b4',
                 label='ROC Kurve (Integral = {:.4f})'.format(auc), lw=1.7)
        #make the plot nicer
        plt.xlim([0.0, 1.1])
        plt.ylim([0.0, 1.1])
        
        plt.xlabel('Signaleffizienz')
        plt.ylabel('Untergrundablehnung')
        if self.variables:
            plt.title(self.variables, loc='left')
        plt.grid(True)
        plt.title(self.name, loc='center')
        # plt.title('CMS Private Work', loc='right')
        plt.legend(loc='best')
        
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()
            
    def _plot_loss(self, train_loss):
        """Plot loss of training and validation data.
        """
        
        plt.plot(train_loss, label= 'Trainingsfehler', color='#1f77b4', ls='',
                 marker='^')
        plt.xlabel('Epoche')
        plt.ylabel('Fehlerfunktion')
        
        if self.variables:
            plt.title(self.variables, loc='left')
        plt.title(self.name, loc='center')
        # plt.title('CMS Private Work', loc='right')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        plt.legend(loc=0)
        
        plt_name = self.name + '_loss'
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()

    def _plot_auc_dev(self, train_auc, val_auc, stop):
        """Plot ROC-AUC-Score development
        """
        plt.plot(range(1, len(train_auc)+1), train_auc, color='#1f77b4', label='Training', ls='',
                 marker='^')
        plt.plot(range(1, len(val_auc)+1), val_auc, color='#ff7f0e', label='Validierung', ls='',
                 marker='^')
        
        # make plot nicer
        plt.xlabel('Epoche')
        plt.ylabel('ROC Integral')
        plt.axvline(x=stop, color='r')
        if self.variables:
            plt.title(self.variables, loc='left')
        plt.title(self.name, loc='center')
        # plt.title('CMS Private Work', loc='right')
        plt.legend(loc='best', frameon=False)

        # save plot
        plt_name = self.name + '_auc_dev'
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()

    def plot_nn(self, branches):
        """Plots the neural network with GraphViz. Not very useful :(
        """
        weight_graph = tf.Graph()
        with weight_graph.as_default():
            weights, biases = self._get_parameters()
            saver = tf.train.Saver()
        with tf.Session(graph=weight_graph) as sess:
            saver.restore(sess, self.savedir + '/{}.ckpt'.format(self.name))
            w = []
            b = []
            for weight, bias in zip(weights, biases):
                w.append(sess.run(weight))
                b.append(sess.run(bias))
                plot_nn = PlotNN(branches, w, b)
                plot_nn.render(self.savedir)
