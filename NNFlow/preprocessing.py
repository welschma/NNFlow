from __future__ import absolute_import, division, print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from root_numpy import root2array
import time
from threading import Thread
import Queue

def root_to_array_helper(files, treename=None, branches=None):
    print('*** Start conversion of ROOT tuple to numpy array ***')
    print ('Starting with tree: {}'.format(treename))
    arr = root2array(files, treename, branches)
    print('*** End conversion of ROOT tuple to numpy array ***')
    print ('Finished tree: {}'.format(treename))



def convert_root_to_array_helper(save_path, name, files, treename=None, branches=None):
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
    # Create empty thread list and queue
    threads = list()
    que = Queue.Queue()

    # Create list to save numpy arrays
    array_list =  list()

    # Check if treename is list, convert if it is a string
    if isinstance(treename, basestring):
        treename = [treename]
    if not isinstance(treename, (list, tuple)):
        print('You have to provide a list of treenames instead of {}'.format(type(treename)))

    for tree in treename:
        t = Thread(target=lambda q, arg1, arg2, arg3: q.put(root_to_array_helper(arg1, arg2, arg3)), args=(que, files, tree, branches), name=tree)
        threads.append(t)
        t.start()

    while(threads):
        for tree_thread in threads:
            if tree_thread.isAlive():
                print('Thread for processing tree {} is alive.'.format(tree_thread.getName()))
                time.sleep(0.1)
            else:
                print('Thread for processing tree {} is dead / finished.'.format(tree_thread.getName()))
                #Get return value for thread
                tree_thread.join()
                print('Joined array for tree {}.'.format(tree_thread.getName()))
                # Remove thread from thread list
                threads.remove(tree_thread)

    # Collect results from all threads
    while not que.empty():
        tmp_array = que.get()
        array_list.append(tmp_array)

    return array_list




def convert_root_to_array(save_path, name, files, treename=None, branches=None, threadFiles=-1, compressFile=True):
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
    threadFiles (int, optional (default=-1)):
    Create a thread for each N files
    compressFile (bool, optional (default=True)):
    Compressing result numpy byte file
    """
    # Create list to save numpy arrays
    array_list =  list()


    if(threadFiles ==-1):
        array_list = convert_root_to_array_helper(save_path, name, files, treename, branches)
    else:
        # Help function to create chunks
        def chunks(l, n):
            """ 
            Yield successive n-sized chunks from l.
            """
            for i in xrange(0, len(l), n):
                yield l[i:i+n]

        file_list = chunks(files, threadFiles)
	for file_entry in file_list:
            print('*** Start processing files: {} ***'.format(file_entry))
            tmp_array_list = convert_root_to_array_helper(save_path, name, file_entry, treename, branches)
            print('*** End processing files: {} ***'.format(file_entry))
            array_list.extend(tmp_array_list)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    np_file = save_path + '/' + name
    if (compressFile):
        np.savez_compressed(np_file, *array_list)
    else:
        np.savez(np_file, *array_list)


class GetVariables:
    """Takes numpy structured arrays and keeps only events of a certain
    category.

    Attributes
    ----------
    category : str
    The category events have to belong to in order to be kept.
    branchlist : str
    Path to the text file with the branches which should be used.
    variable list(str) :
    A list cotaining filled with the branches from branchlist.
    save_path : str
    Path to the directory the processed array will be saved to.
    arr_name : str
    A name for the new array.
    """
    
    def __init__(self, variablelist, weightlist, category, save_in):
        """Initializes the class with the given attributes.
        """
        self._category = category
        # create dir where arrays will be saved in
        self.save_in = save_in  + '/' + category
        if not os.path.isdir(self.save_in):
            os.makedirs(self.save_in)
            print('created directory {}'.format(self.save_in))
            
        self._vars = variablelist
        self._weights = weightlist

    def run(self, sig_paths, bkg_paths):

        # load structured arrays
        structured_sig = self._load_array(sig_paths)
        structured_bkg = self._load_array(bkg_paths)

        # get all variables from self._vars from the structured array as an
        # 2d np.array
        sig = self._get_vars(structured_sig, self._vars)
        bkg = self._get_vars(structured_bkg, self._vars)
        
        sig['weights'] = self._get_unnormalized_weights(structured_sig)
        bkg['weights'] = self._get_unnormalized_weights(structured_bkg)

        sig = self._get_category(sig, structured_sig)
        bkg = self._get_category(bkg, structured_bkg)
        
        n_sig_events = sig['data'].shape[0]
        n_bkg_events = bkg['data'].shape[0]
        sig['labels'] = self._get_labels(n_sig_events, 1.0)
        bkg['labels'] = self._get_labels(n_bkg_events, 0.0)

        self._save_array(sig, bkg)

    def _load_array(self, path_list):
        """Loads all structured arrays given in the path list and stacks them
        along axis 0.
        """
        array_list = []
        for path in path_list:
            array = np.load(path)
            array_list.append(array)
        array = np.concatenate(array_list, axis=0)

        return array
        
    def _get_vars(self, structured_array, var_list):
        """Get _vars out of the structured array and place them into a 
        normal numpy ndarray. If the branch is vector like, only keep the first
        four entries (jet variables).
        """

        # define vector like variables
        jets = ['CSV', 'Jet_CSV', 'Jet_CosThetaStar_Lepton', 'Jet_CosTheta_cm',
                'Jet_Deta_Jet1', 'Jet_Deta_Jet2','Jet_Deta_Jet3',
                'Jet_Deta_Jet4','Jet_E','Jet_Eta','Jet_Flav','Jet_GenJet_Eta',
                'Jet_GenJet_Pt', 'Jet_M','Jet_PartonFlav', 'Jet_Phi',
                'Jet_PileUpID', 'Jet_Pt']

        array_list = []
        vars = []
        
        for var in var_list:
            if var in jets:
                # only keep the first four entries of the jet vector
                array = [jet[:4] for jet in structured_array[var]]
                array_list.append(np.vstack(array))
                vars += [var+'_{}'.format(i) for i in range(1,5)]
            else:
                array = structured_array[var].reshape(-1,1)
                array_list.append(array)
                vars += [var]

        data_dict = {'data': np.hstack(array_list), 'vars': vars}
        return data_dict

    def _get_unnormalized_weights(self, structured_array):
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
        weights = self._get_vars(structured_array, self._weights)
        weights = np.prod(weights['data'], axis=1).reshape(-1,1)
        
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

            if self._check_category(N_LL, N_TL, N_J, N_BTM, self._category):
                keep_events.append(event)
            else:
                continue

        keep_dict = {'data': data_dict['data'][keep_events],
                     'weights': data_dict['weights'][keep_events],
                     'vars': data_dict['vars']}

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
                    '62': (N_LL == 1 and N_TL == 1 and N_J >= 6 and N_BTM >= 2),
                    '63': (N_LL == 1 and N_TL == 1 and N_J >= 6 and N_BTM == 3),
                    '64': (N_LL == 1 and N_TL == 1 and N_J >= 6 and N_BTM >= 4),
                    'all': True}

        return category[name]

    def _split_array(self, array):
        num_evts = array.shape[0]
        num_train_evts = int(0.5*num_evts)
        num_val_evts = int(0.1*num_evts)
        # split array in [train, val, test]
        arrays = np.split(array, [num_train_evts,
                                 (num_train_evts+num_val_evts)])
        
        # normalize weights for each array
        
        for array in arrays:
            array[:, -1] /= np.sum(array[:, -1])

        return arrays
        
    
    def _save_array(self, sig, bkg):
        """Stacks data and saves the array to given path.

        Arguments
        ---------
        sig_dict : dict
        Dictionary containing signal events.
        bg_dict : dict
        Dictionary containing background events.
        """
         # write variable names to file
        with open(self.save_in + '/vars.txt', 'w') as f:
            for var in sig['vars']:
                f.write(var + '\n')
                
        sig = np.hstack((sig['labels'], sig['data'], sig['weights']))
        bkg = np.hstack((bkg['labels'], bkg['data'], bkg['weights']))

        sig = self._split_array(sig)
        bkg = self._split_array(bkg)

        np.save(self.save_in + '/train.npy', np.vstack((sig[0], bkg[0])))
        np.save(self.save_in + '/val.npy', np.vstack((sig[1], bkg[1])))
        np.save(self.save_in + '/test.npy', np.vstack((sig[2], bkg[2])))
