# A Binary Multilayerperceptron Classifier. Currently Depends on a custom
# dataset class defined in data_frame.py.
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from sklearn.metrics import roc_auc_score, roc_curve
from .draw_nn import PlotNN

class BinaryMLP:
    """A Binary Classifier using a Multilayerperceptron.

    Makes probability predictions on a set of features (A 1-dimensional
    numpy vector belonging either to the 'signal' or the 'background'.

    """

    def __init__(self, n_features, h_layers, savedir, activation='relu'):
        """Initializes the Classifier.

        Arguments:
        ----------------
        nfeatures (int) :
        The number of input features.
        hlayers (list):
        A list representing the hidden layers. Each entry gives the number of
        neurons in the equivalent layer.
        savedir (str) :
        Path to the directory the model should be saved in.
        activation (str) :
        Default is 'relu'. Activation function used in the model. Also possible 
        is 'tanh' or 'sigmoid'.

        Attributes:
        ----------------
        name (str) :
        Name of the model.
        savedir (str) :
        Path to directory everything will be saved in.
        trained (bool) : 
        Flag wether model has been trained or not
        """
        self.n_features = n_features
        self.n_labels = 1
        self.h_layers = h_layers
        self.activation = activation
        self.name = savedir.rsplit('/')[-1]
        self.variables = savedir.rsplit('/')[1]
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
        n_features = self.n_features
        h_layers = self.h_layers

        weights = [tf.Variable(
            tf.random_normal(shape = [n_features, h_layers[0]],
                             stddev = tf.sqrt(2.0/n_features)), name = 'W_1')]
        biases = [tf.Variable(tf.fill(dims=[h_layers[0]],
                                      value=0.1), name = 'B_1')]

        # fancy loop in order to create weights connecting the hidden layer.
        if len(h_layers) > 1:
            for i in range(1, len(h_layers)):
                weights.append(tf.Variable(tf.random_normal(
                    shape = [h_layers[i-1], h_layers[i]],
                    stddev = tf.sqrt(2.0/h_layers[i-1])),
                                           name = 'W_{}'.format(i+1)))
                biases.append(tf.Variable(tf.fill(dims=[h_layers[i]],
                                                  value=0.1),
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
    
    def _model(self, data, W, B, keep_prob=1.0):
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
            tf.add(tf.matmul(data, W[0]), B[0])), keep_prob)
        
        # fancy loop for creating hidden layer
        if len(self.h_layers) > 1:
            for weight, bias in zip(W[1:-1], B[1:-1]):
                layer = tf.nn.dropout(activation(
                    tf.add(tf.matmul(layer, weight), bias)),keep_prob)

        output_node = tf.nn.sigmoid(tf.add(tf.matmul(layer, W[-1]), B[-1]),
                            name='output_node')

        return output_node

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
            x = tf.placeholder(tf.float32, [None, self.n_features])
            y = tf.placeholder(tf.float32, [None, 1])
            w = tf.placeholder(tf.float32, [None, 1])

            x_mean = tf.Variable(np.mean(train_data.x, axis=0).astype(np.float32),
                                 trainable=False, name='x_mean')
            x_std = tf.Variable(np.std(train_data.x, axis=0).astype(np.float32),
                                trainable=False, name='x_std')

            x_scaled = tf.div(tf.sub(x, x_mean), x_std)
            
            weights, biases = self._get_parameters()

            #prediction
            y_ = self._model(x_scaled, weights, biases, keep_prob)
            yy_ = self._model(x_scaled, weights, biases)

            # loss function, added small number for more numerical stability
            xentropy = -(tf.mul(y, tf.log(y_ + 1e-10)) +
                         tf.mul(1-y, tf.log(1-y_ + 1e-10)))
            l2_reg = beta*self._l2_regularization(weights)
            loss = tf.reduce_mean(tf.mul(w,xentropy)) + l2_reg

            # optimizer
            opt, global_step = self._get_optimizer()
            train_step = opt.minimize(loss, global_step=global_step)

            # initialize the variables
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(weights + biases + [x_mean, x_std])

        train_start = time.time()

        with tf.Session(graph=train_graph) as sess:
            self.model_loc = self.savedir + '/{}.ckpt'.format(self.name)
            sess.run(init)
            
            train_auc = []
            val_auc = []
            train_losses = []
            val_losses = []
            early_stopping  = {'auc': 0.0, 'epoch': 0}
             
            print(125*'-')
            print('Train model: {}'.format(self.model_loc))
            print(125*'-')
            print('{:^20} | {:^20} | {:^20} | {:^20} |{:^25}'.format(
                'Epoch', 'Training Loss', 'Validation Loss',
                'AUC Training Score', 'AUC Validation Score'))
            print(125*'-')

            for epoch in range(epochs):
                total_batches = int(train_data.n/batch_size)
                train_loss = 0
                
                for _ in range(total_batches):
                    train_x, train_y, train_w= train_data.next_batch(batch_size)

                    _,  = sess.run([train_step],
                                 {x: train_x, y: train_y, w: train_w})
                
                # monitor training
                val_losses.append(sess.run(loss, {x: val_data.x,
                                                  y: val_data.y,
                                                  w: val_data.w}))
                train_losses.append(sess.run(loss, {x: train_data.x,
                                                    y: train_data.y,
                                                    w: train_data.w}))
                train_pre = sess.run(yy_, {x : train_data.x})
                train_auc.append(roc_auc_score(train_data.y, train_pre))
                val_pre = sess.run(yy_, {x : val_data.x})
                val_auc.append(roc_auc_score(val_data.y, val_pre))
                
                print('{:^20} | {:^20.4e} | {:^20.4e} | {:^20.4f} | {:^25.4f}'
                      .format(epoch+1, train_losses[-1], val_losses[-1],
                              train_auc[-1], val_auc[-1]))

                if early_stop:
                    # check for early stopping, only save model if val_auc has
                    # increased
                    if val_auc[-1] > early_stopping['auc']:
                        save_path = saver.save(sess, self.model_loc)
                        early_stopping['auc'] = val_auc[-1]
                        early_stopping['epoch'] = epoch+1
                        early_stopping['val_pre'] = val_pre
                    elif (epoch - early_stopping['epoch']) > early_stop:
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
                train_data.shuffle()
                
            train_end = time.time()
            
            print(125*'-')
            self._validation(early_stopping['val_pre'], val_data.y)
            self._plot_auc_dev(train_auc, val_auc, epochs)
            self._plot_loss(train_losses, val_losses)
            self.trained = True
            self._write_parameters(batch_size, keep_prob, beta,
                                   (train_end - train_start), early_stopping)
            print('Model saved in: {}'.format(save_path))
            print(125*'-')
            
    def _l2_regularization(self, weights):
        """Calculate and adds the squared values of the weights. This is used
        for L2 Regularization.
        """
        weights = map(lambda x: tf.nn.l2_loss(x), weights)

        return tf.add_n(weights)

    def _write_parameters(self, batch_size, keep_prob, beta, time,
                          early_stop):
        """Writes network parameters in a .txt file
        """

        with open('{}/NN_Info.txt'.format(self.savedir), 'w') as f:
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
            f.write('Validation auc score: {}\n'.format(early_stop['auc']))
            f.write('Batch Size: {}\n'.format(batch_size))
            f.write('Dropout: {}\n'.format(keep_prob))
            f.write('L2 Regularization: {}\n'.format(beta))
            f.write('Training Time: {} s\n'.format(time))

    def _validation(self, pred, labels):
        """Validation of the training process.
        Makes plots of ROC curves and displays the development of AUC score.

        Arguments:
        ----------------
        pred (np.array, shape(-1,)) :
        Predictions for data put in the model.
        labels (np.array, shape(-1)) :
        Lables of the validation dataset.

        Returns:
        ----------
        auc_sore (float):
        Number between 0 and 1.0. Area under the ROC curve. 
        Displays the model's quality.
        """

        # distribution
        y = np.hstack((pred, labels))
        sig = y[y[:,1]==1, 0]
        bg = y[y[:,1]==0, 0]
       
        bin_edges = np.linspace(0, 1, 30)
        plt.hist(sig, bins=bin_edges, color='#1f77b4',histtype='step',
                 label='Signal', normed='True', lw=1.7)
        bins, _, _ = plt.hist(bg, bins=bin_edges, color='#d62728',
                              histtype='step', label='Untergrund',
                              normed='True', lw=1.7)
        

        plt.legend(loc='upper left', frameon=False)
        plt.xlabel('Netzwerk Ausgabe')
        plt.ylabel('normierte Ereignisse')
        plt.title(self.variables, loc='left')
        plt.title(self.name, loc='center')
        plt.title('CMS Private Work', loc='right')
        
        
        plt_name = self.name + '_dist'
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()

        # roc curve
        pred = np.reshape(pred, -1)
        fpr, tpr, thresh = roc_curve(labels, pred)
        auc = roc_auc_score(labels, pred)
        #plot the roc_curve
        plt_name = self.name +  '_roc'
        
        plt.plot(tpr, np.ones(len(fpr)) - fpr, color='#1f77b4',
                 label='ROC Kurve (Integral = {:.4f})'.format(auc), lw=1.7)
        #make the plot nicer
        plt.xlim([0.0, 1.1])
        plt.ylim([0.0, 1.1])
        
        plt.xlabel('Signaleffizienz')
        plt.ylabel('Untergrundablehnung')
        plt.title(self.variables, loc='left')
        plt.title(self.name, loc='center')
        plt.title('CMS Private Work', loc='right')
        plt.legend(loc='best', frameon=False)
        
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()

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
            x = tf.placeholder(tf.float32, [None, self.n_features])
            x_mean = tf.Variable(-1.0, validate_shape=False,  name='x_mean')
            x_std = tf.Variable(-1.0, validate_shape=False,  name='x_std')

            x_scaled = tf.div(tf.sub(x, x_mean), x_std)
            
            y_prob = self._model(x_scaled, weights, biases)
            
            saver = tf.train.Saver()
        with tf.Session(graph = predict_graph) as sess:
            saver.restore(sess, self.savedir + '/{}.ckpt'.format(self.name))
            prob = sess.run(y_prob, {x: data})

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
            x = tf.constant(-1.0, shape=[1, self.n_features],
                               name='input_node')
            x_mean = tf.Variable(-1.0, validate_shape=False,  name='x_mean')
            x_std = tf.Variable(-1.0, validate_shape=False,  name='x_std')

            x_scaled = tf.div(tf.sub(x, x_mean), x_std, name='x_scaled')
            
            y= self._model(x_scaled, weights, biases)
            
            saver = tf.train.Saver()
        with tf.Session(graph=export_graph) as sess:
            saver.restore(sess, self.savedir + '/{}.ckpt'.format(self.name))
            const_graph=tf.graph_util.convert_variables_to_constants(
                sess, export_graph.as_graph_def(), ['output_node'])
            tf.train.write_graph(const_graph, self.savedir, self.name + ".pb",
                                 as_text=False)

            
    def _plot_loss(self, train_loss, val_loss):
        """Plot loss of training and validation data.
        """
        train_loss = np.array(train_loss)*10**6
        val_loss = np.array(val_loss)*10**6
        
        plt.plot(train_loss, label= 'Trainingsfehler', color='#1f77b4', lw=1.7)
        plt.plot(val_loss, label= 'Validierungsfehler', color='#ff7f0e', lw=1.7)
        plt.xlabel('Epoche')
        plt.ylabel('Fehlerfunktion (mult. mit 10^6)')
        
        plt.title(self.variables, loc='left')
        plt.title(self.name, loc='center')
        plt.title('CMS Private Work', loc='right')
        # plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        plt.legend(loc=0, frameon=False)
        
        plt_name = self.name + '_loss'
        plt.savefig(self.savedir + '/' + plt_name + '.pdf')
        plt.savefig(self.savedir + '/' + plt_name + '.png')
        plt.savefig(self.savedir + '/' + plt_name + '.eps')
        plt.clf()

    def _plot_auc_dev(self, train_auc, val_auc, nepochs):
        """Plot ROC-AUC-Score development
        """
        plt.plot(train_auc, color='#1f77b4', label='Training', lw=1.7)
        plt.plot(val_auc, color='#ff7f0e', label='Validierung', lw=1.7)
        
        # make plot nicer
        plt.xlabel('Epoche')
        plt.ylabel('ROC Integral')
        plt.title(self.variables, loc='left')
        plt.title(self.name, loc='center')
        plt.title('CMS Private Work', loc='right')
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
