# NNFlow
A simple interface for Google's TensorFlow 

## Basic Workflow
### Converting Root nTuples into numpy arrays
NNFlow provides a function, based on root-numpy, for converting Root nTuples into numpy arrays.
Example:
```
from NNFlow.preprocessing import convert_root_to_array

ntuples = ['path/to/ntuple_1', 'path/to/ntuple_2', ...]
save_to = 'path/to/storage/directory'
name = 'array.npy'
my_tree = 'tree'
my_branches = ['branch_1', 'branch_2']

convert_root_to_array(save_to, name, ntuples, tree=my_tree, branches=my_branches, threadFiles=-1, compressFile=True)
```
* If there is only one TTree in the nTuple you do not have to provide the name of the tree which should be converted.
* If you want to convert all branches of a tree, do not provide a branchlist.
* NOTE: The array the data is saved to is of type np.structure_array (https://docs.scipy.org/doc/numpy/user/basics.rec.html) while several arrays are saved into a single file in a (un)compressed .npz format.
* The preprocessing is making use of threading by default if you give more than one branch name.
* Furthermore, you can make use of threading for splitting the given files into thread by defining the number of files per thread via ```threadFiles```. Here, ```threadFiles=-1``` results in no additional threading.
* In addition, you can define if you would like to have a compressed numpy byte array file or an uncompressed file via the flag ```compressFile```.

### Preprocess numpy arrays
Once you've converted the nTuples to numpy arrays, you have do some other preprocessing steps, like getting some specific branches from the array if you've converted everything in the nTuple.
Example:
```
from NNFlow.preprocessing import GetBranches

ttH = 'path/to/ttH.npy'
ttbarSL = 'path/to/ttbarSL.npy'
branches = 'branchlist.txt'
weights = ['weight_1', 'weight_2', 'weight_3', ...]
category = '63'

gb = GetBranches('my_variables', category, branches, weights)
gb.process(ttH, ttbarSL, 'even')
```
This script gets the branches defined in the ```branchlist.txt``` file (one branch name in every line) from the structured array and saves them into a normal 2D np.ndarray.
If a branch is vector like then you should check if the branch is included in the list in line 116 in file [NNFlow/preprocessing](NNFlow/preprocessing.py) (checking for vector like data is not automated yet).
The signal (ttH) and backgroudn (ttbarSL) are combined into one array.
Labels for signal (1) and background (0) are added in the first column, the weights are added in the last column.
The event weights are normed to sum 1 respectively for signal and background.
The directory 'my_variables/category' is created and the array 'even.npy' is save in it.

### Train a neural network
Here is a simple example script if you want to train a neural network.
Please have a look at [NNFlow/binary_mlp](NNFlow/binary_mlp.py) and [NNFlow/data_frame](NNFlow/data_frame.py) files for more information about the options.
```
from NNFlow.binary_mlp import BinaryMLP
from NNFlow.data_frame import load_data

even = 'my_variables/category/even.npy'
odd = 'my_variables/category/odd.npy'
train, val, test = load_data(even, odd)

save_model_to = 'my_variables/category/models/2x100'
n_features = train.nfeatures
hidden_layers = [100, 100] 

# create neural net
mlp = BinaryMLP(train.nfeatures, [100, 100], save_model_to)

# train 
mlp.train(train, val, epochs=250, lr=1e-3)

# classify new events
y_test = mlp.classify(test.x)
```
This script will train a neural network with two hidden layers with 100 neurons.
The network is saved to the directory 'save_model_to'. 
Also some controll plots of the training process will be saved there.

## Dependencies
Tested with
* tensorflow (0.12.0rc1)
* numpy (1.11.2)
* matplotlib (1.5.3)
* scikit-learn (0.18.1)
* root-numpy (4.5.2)
* graphviz (0.5.1)

