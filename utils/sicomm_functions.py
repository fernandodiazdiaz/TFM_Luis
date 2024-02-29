import numpy as np
import pandas as pd
import networkx as nx
import scipy.linalg as la






def EB_index(G):   
    
    ### find Estrada-Benzi balance index of a network    
       
    # obtain adjacency matrix if needed
    if type(G) ==  nx.classes.graph.Graph or type(G) == nx.classes.digraph.DiGraph:
        A = nx.adjacency_matrix(G).todense()
    elif type(G) == np.ndarray or type(G) == np.matrix:
        A = G

    # calculate signed communicability
    A0 = np.abs(A)
    Comm = la.expm(A)
    Comm0 = la.expm(A0)
    
    # calculate balance
    K = np.trace(Comm)/np.trace(Comm0)       # node balance
    
    return K




def communicability_metrics(G, nodelist = None, output = 'all'):
    
    # computes the communicabaility metrics of a given network.
    # Input: 
    # G: networkx Graph object.
    # nodelist (default: None). The rows and columns are ordered according to the nodes in nodelist. If nodelist is None, then the ordering is produced by G.nodes().
    # output ('all','comm','distance','angle'): the communicability metric we want to compute.
    # Output:
    # S: communicability matrix.
    # xi: communicability distance matrix.
    # theta: communicability angle matrix.
    # X: matrix whose column are position vectors of the nodes in communicability space

    N = G.order()
    if nodelist == None: nodelist = G.nodes()

    # compute communicability metrics
    A = nx.adjacency_matrix(G, nodelist = nodelist).todense()

    if output != 'comm_coordinates':  # compute communicability only if needed
        S = la.expm(A)    # signed communicability
        K = np.diag(S)    # node balance

    if output == 'all' or output == 'comm_coordinates':
        Lamb, U = la.eig(A)
        Lamb, U = np.real(Lamb), np.real(U)  # remove imaginary parts due to numerical errors
        X = np.diag(np.exp(Lamb/2))@U.T
    
    if output == 'all' or output == 'distance':
        xi2 = np.outer(K,np.ones(N))+ np.outer(np.ones(N), K) - 2*S  #squared comm distance matrix
        xi = np.where(xi2 < 0, 0, np.sqrt(xi2))  #comm distance (correct for numerical errors)
        assert np.isnan(xi).any() == False, 'xi has NaNs'

    if output == 'all' or output == 'angle':
        cos_theta = S/np.sqrt(np.outer(np.ones(N), K)*np.outer(K, np.ones(N)))
        cos_theta[cos_theta>1] = 1  # correct for numerical errors
        cos_theta[cos_theta<-1] = -1
        theta = np.arccos(cos_theta) # comm angle matrix
    
    # return the desired output
    if output == 'all':
        return S, xi, theta, X
    elif output == 'comm':
        return S
    elif output == 'distance':
        return xi
    elif output == 'angle':
        return theta
    elif output == 'comm_coordinates':
        return X
    else:
        raise ValueError(f"output must be 'all', 'comm', 'distance', 'angle' or 'comm_coordinates'")






def find_allies(G, name, criterion, verbose = True):
    # returns the main allies of a given person using acommunicability-based method
    # Input:
    # G: a nx.Graph object containing the signed network to analyze
    # name: name of the node that we want to analyze
    # criterion: name of the criterion:
    #  - criterion = 'comm' returns the nodes ordered by decreasing communicability
    #  - criterion = 'distance' returns the nodes ordered by increasing communicability distance
    #  - criterion = 'angle' returns the nodes ordered by increasing communicability angle
    # verbose (default: True): if True, prints the allies
    # output:
    # ordered_allies: pandas Series containing the allies in decreasing importance

    # create network and compute metrics
    nodelist = G.nodes()

        
    if criterion == 'comm':
        variable = communicability_metrics(G, nodelist, output='comm')
    elif criterion == 'distance':
        variable = communicability_metrics(G, nodelist, output='distance')
    elif criterion == 'angle':
        variable = communicability_metrics(G, nodelist, output='angle')
    else:
        raise ValueError(f"The only valid criteria are 'comm', 'distance' or 'angle'")

    # order the column corresponding to the given node
    df = pd.DataFrame(variable, index = nodelist, columns = nodelist)
    if criterion == 'comm': # communicability in descending order
        ordered_allies = df[name].sort_values(ascending = False)
    else:                   # distance and angles in ascending order
        ordered_allies = df[name].sort_values(ascending = True)

    if verbose:
        print(f'Top allies of {name} (criterion: {criterion}):')
        print(ordered_allies)

    return ordered_allies
      









