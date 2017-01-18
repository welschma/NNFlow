from __future__ import absolute_import, division, print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from root_numpy import root2array


def convert_root_to_array(save_path, name, files, treename=None, branches=None):
    """Convert trees in ROOT files into a numpy structured array.

    Arguments
    ---------
    save_path (str):
    Path to the directory the array will be saved in.
    files (str or list(str)):
    The name of the files that will be converted into one structured array.
    treename (str, optional (default=None)):
    Name of the tree to convert.
    branches (str or list(str), optional (default=None)):
    List of branch names to include as collumns of the array. If None all
    branches will be included.    
    """
    arr = root2array(files, treename, branches)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    np_file = save_path + '/' + name
    np.save(npfile, arr)


class GetBranches:
    """Takes numpy structured arrays and keeps only events of a certain
    category.

    Attributes
    ----------
    category : str
    The category events have to belong to in order to be kept.
    branchlist : str
    Path to the text file with the branches which should be used.
    branches : list(str)
    A list filled with the branches from branchlist.
    save_path : str
    Path to the directory the processed array will be saved to.
    arr_name : str
    A name for the new array.
    """
    
    def __init__(self, category, branchlist, weightlist):
        """Initializes the class with the given attributes.
        """
        
        self.category = category
        self.save_path = branchlist.split('/')[-1].split('.')[0] + '/' + category

        with open (branchlist, 'r') as f:
            self.branches = [line.strip('\n') for line in f]
        self.weights = weightlist

    def process(self, signal_path, background_path, arr_name):
        self.arr_name = arr_name
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
            print('created directory {}'.format(self.save_path))
        
        structured_sig = np.load(signal_path)
        structured_bg = np.load(background_path)
        sig_data, sig_branches = self._get_branches(structured_sig,
                                                    self.branches)
        bg_data, bg_branches = self._get_branches(structured_bg,
                                                    self.branches)

        self.new_branches = sig_branches
        self._controll_plot(sig_data, bg_data, sig_branches)

        sig = {'data': sig_data}
        bg = {'data': bg_data}

        sig['weights'] = self._get_weight(structured_sig)
        bg['weights'] = self._get_weight(structured_bg)

        sig = self._get_category(sig, structured_sig)
        bg = self._get_category(bg, structured_bg)
        
        n_sig_events = sig['data'].shape[0]
        n_bg_events = bg['data'].shape[0]

        sig['labels'] = self._get_labels(n_sig_events, 1.0)
        bg['labels'] = self._get_labels(n_bg_events, 0.0)

        self._save_array(sig, bg)

    def _get_branches(self, structured_array, branches):
        """Get branches out of the structured array and place them into a 
        normal numpy ndarray. If the branch is vector like, only keep the first
        four entries (jet variables).

        Arguments
        ---------
        structured_array : numpy structured array
        Structured array converted from ROOT file.
        branches : list(str)
        List of branches to out of the structured array.

        Returns
        -------
        ndarray : numpy ndarray
        An array filled with the data.
        new_branches: list(str)
        List of variables which ndarray is filled with. Each entry represents
        the corresponding column of the array.
        """

        # define vector like variables
        jets = ['CSV', 'Jet_CSV', 'Jet_CosThetaStar_Lepton', 'Jet_CosTheta_cm',
                'Jet_Deta_Jet1', 'Jet_Deta_Jet2','Jet_Deta_Jet3',
                'Jet_Deta_Jet4','Jet_E','Jet_Eta','Jet_Flav','Jet_GenJet_Eta',
                'Jet_GenJet_Pt', 'Jet_M','Jet_PartonFlav', 'Jet_Phi',
                'Jet_PileUpID', 'Jet_Pt']

        ndarray = []
        new_branches = []
        
        for branch in branches:
            if branch in jets:
                # only keep the first four entries of the jet vector
                array = [jet[:4] for jet in structured_array[branch]]
                ndarray.append(np.vstack(array))
                new_branches += [branch+'_{}'.format(i) for i in range(1,5)]
            else:
                array = structured_array[branch].reshape(-1,1)
                ndarray.append(array)
                new_branches += [branch]
        
        return np.hstack(ndarray), new_branches

    def _get_weight(self, structured_array):
        """Calculate the weight for eacht event.

        For each event we weill calculate:
        Weight_XS * Weight_CSV * Weight_pu69p2
        Then, the weights are normalized, so that the sum over all weights is 
        equal to 1.

        Arguments
        ---------
        structured_array : numpy structured array
        Structured array converted from ROOT file.

        Returns
        -------
        weights : numpy ndarray
        An array of shape (-1,1) filled with the weight of each event.
        """
        # weight_names = ['Weight_XS', 'Weight_CSV', 'Weight_pu69p2']
        # weight_names = ['Weight_XS']

        weights, _ = self._get_branches(structured_array, self.weights)
        weights = np.prod(weights, axis=1).reshape(-1,1)
        weights /= np.sum(weights)
        
        return weights
        
    def _get_labels(self, n_events, label):
        """Create labels.
        
        Arguments
        ---------
        n_events : int
        Number labels to create.
        label : float
        Label.
        
        Returns
        -------
        label : numpy ndarray
        A numpy ndarray of shape (n_events, 1) filled with the label.
        """

        labels = np.full(shape=(n_events, 1), fill_value=label)
        return labels
    
    def _get_category(self, data_dict, structured_array):
        """Checks if the data belongs to the given category. Only keep events
        that do.

        Arguments
        ---------
        data_dict : dict
        Dictionary filled with event variables and corresponding weights.
        structured_array : numpy structured array
        Structured array converted from ROOT file.
        """
        
        keep_events = []   
        for event in range(structured_array.shape[0]):
            N_LL = structured_array[event]['N_LooseLeptons']
            N_TL = structured_array[event]['N_TightLeptons']
            N_J = structured_array[event]['N_Jets']
            N_BTM = structured_array[event]['N_BTagsM']

            if self._check_category(N_LL, N_TL, N_J, N_BTM, self.category):
                keep_events.append(event)
            else:
                continue

        keep_dict = {'data': data_dict['data'][keep_events],
                     'weights': data_dict['weights'][keep_events]}

        return keep_dict
    
    def _check_category(self, N_LL, N_TL, N_J, N_BTM, name):
        """Returns category bool.

        Arguments:
        ----------------
        N_LL  (int): N_LooseLeptons
        N_TL  (int): N_TightLeptons
        N_J   (int): N_Jets
        N_BTM (int): N_BTagsM]
        """
        category = {'43': (N_LL == 1 and N_TL == 1 and N_J == 4 and N_BTM == 3),
                    '44': (N_LL == 1 and N_TL == 1 and N_J == 4 and N_BTM == 4),
                    '53': (N_LL == 1 and N_TL == 1 and N_J == 5 and N_BTM == 3),
                    '54': (N_LL == 1 and N_TL == 1 and N_J == 5 and N_BTM >= 4),
                    '62': (N_LL == 1 and N_TL == 1 and N_J >= 6 and N_BTM == 2),
                    '63': (N_LL == 1 and N_TL == 1 and N_J >= 6 and N_BTM == 3),
                    '64': (N_LL == 1 and N_TL == 1 and N_J >= 6 and N_BTM >= 4),
                    'all': True}

        return category[name]

    def _save_array(self, sig, bg):
        """Stacks data and saves the array to given path.

        Arguments
        ---------
        sig_dict : dict
        Dictionary containing signal events.
        bg_dict : dict
        Dictionary containing background events.
        """
        array_dir = self.save_path + '/' + self.arr_name
        print('saving array to: {}, '.format(array_dir), end='')
        sig_arr = np.hstack((sig['labels'], sig['data'], sig['weights']))
        bg_arr = np.hstack((bg['labels'], bg['data'], bg['weights']))

        ndarray = np.vstack((sig_arr, bg_arr))
        np.save(array_dir + '.npy', ndarray)
       
        with open(self.save_path + '/branches.txt', 'w') as f:
            for branch in self.new_branches:
                f.write(branch + '\n')
        
        print('done.')

    def _controll_plot(self, sig, bg, branches):
        """Plot histograms of all variables

        Arguments
        ---------
        sig : numpy ndarray
        Array containing signal data.
        bg : numpy ndarray
        Array containing background data.
        branches : list(str)
        List of variable names.
        """

        plot_dir = self.save_path + '/controll_plots/' + self.arr_name
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        for variable in range(sig.shape[1]):
            # get binedges
            sig_min, bg_min = np.amin(sig[:, variable]), np.amin(bg[:, variable])
            sig_max, bg_max = np.amax(sig[:, variable]), np.amax(bg[:, variable])
            if sig_min < bg_min:
                glob_min = sig_min
            else:
                glob_min = bg_min
            if sig_max > bg_max:
                glob_max = sig_max
            else:
                glob_max = bg_max

            bin_edges = np.linspace(glob_min, glob_max, 30)
                        
            n, bins, _ = plt.hist(sig[:, variable], bins=bin_edges,
                                  histtype='step', normed=True, label='Signal',
                                  color='black')
            n, bins, _ = plt.hist(bg[:, variable], bins=bin_edges,
                                  histtype='step', normed=True,
                                  label='Background', color='red', ls='--')
            plt.ylabel('norm. to unit area')
            plt.xlabel(branches[variable])
            plt.legend(loc='best', frameon=False)
            plt.savefig(plot_dir + '/' + branches[variable] + '.pdf')
            plt.savefig(plot_dir + '/' + branches[variable] + '.png')
            plt.savefig(plot_dir + '/' + branches[variable] + '.eps')
            plt.clf()
                                  
