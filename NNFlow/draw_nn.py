import graphviz as gv
import numpy as np

class PlotNN:
    """Plot a Neural Network with given weights and biases
    with graphviz.
    """

    def __init__(self, inputs, weights, biases):
        self.inputs = inputs
        self.weights = weights
        self.biases = biases
        self.previous_layer = []
        self.previous_bias = []

    def render(self, savedir):
        g = gv.Graph()
        # set some graph attributes
        g.graph_attr['rankdir'] = 'LR'
        g.node_attr['shape'] = 'circle'
        g.node_attr['label'] = ''

        # add input layer
        self._add_input_layer(g)
        self._add_hidden_layer(g)

        g.render('nn_plot', savedir)

    def _add_input_layer(self, graph):
        self.input_labels = ['I' + str(i+1) for i in range(len(self.inputs))]
        for n, l in zip(self.input_labels, self.inputs):
            graph.node(n, xlabel=l, label='')
        self.previous_layer.append(self.input_labels)
        b = 'B_0'
        graph.node(b, label='', shape='none')
        self.previous_bias.append(b)

    def _add_hidden_layer(self, graph):
        for weight, bias, n in zip(self.weights, self.biases,
            [i for i in range(len(self.weights))]):
            hidden = ['H{}_'.format(n+1) + str(i+1) for i in range(weight.shape[1])]
            for h in hidden:
                graph.node(h, label='')

            for i, p in enumerate(self.previous_layer[-1]):
                for j, h in enumerate(hidden):
                    width = weight[i,j]
                    if width > 0:
                        c = 'blue'
                    else:
                        c = 'red'
                    width = np.abs(width)
                    graph.edge(p, h, penwidth=str(width), color=c)

            b = 'B_' + str(n+1)
            graph.node(b, xlabel='Bias')
            graph.edge(self.previous_bias[-1], b, style='invis')
            for i, h in enumerate(hidden):
                width = bias[i]
                if width > 0:
                        c = 'blue'
                else:
                    c = 'red'
                width = np.abs(width)
                graph.edge(b, h, penwidth=str(width), color=c)
            self.previous_layer.append(hidden)
            self.previous_bias.append(b)
