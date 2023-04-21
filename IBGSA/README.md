# Basic Implementation (IBGSA_basic.py)
Implementation of IBGSA feature selector + two choices of basic classifiers. Elitism and normalization of Euclidean distance is included. Results are listed in console.

## Parameters
### 

 - **csv_path** : File path name of the csv file
 - **iterations** : The number of iterations ran
 - **G0** : Initial gravitational constant
 - **alpha** : The weight of force applied to each dimension during each iteration
 - **beta** : The decay rate of the gravitational constant over time
 - **agent_num** : Number of agents used to move around in the search space
 - **classifier** : Choices are 'dt' for decision tree and 'rf' for random forest
 - **show_curve** : Boolean if you want the AUC ROC curve to pop up

## Current State

The file has 2 calls that will give you results for both a decision tree and random forest classifier run, using the basic parameters we used for all our test runs, and with the curve graphics turned off. The calls assume the csv file is located in the same folder as the .py

## To run
Run python IBGSA_basic.py after installing necessary dependencies

# Basic Implementation (IBGSA_paper.py)
Implementation of the pseudocode presented in Sayed et al. 's paper.

## Current State

The file contains two global variables:
- USE_RANDOM_FOREST: If this value is set to false, IBGSA will use decision tree as its classifier, else it will use random forest.
- SHOW_CURVE : If this value is set to false, the AUC-ROC curve will not be displayed at the end of the run.

Hyperparameters to IBGSA can also be modified by changing the function call to IBGSA in line 192.

## To run
Run python IBGSA_paper.py after installing necessary dependencies
