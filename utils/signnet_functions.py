import numpy as np
import pandas as pd
import networkx as nx



def create_network(filepath, nodelist = None):
    # creates a networkx Graph object from a csv file containing an edge array with the edge signs
    # input:
    # filepath: the path to the csv file 
    # subgraph_nodes (default: None): a list of the nodes that should be included in the network. If None, the function includes all nodes in the network.
    
    edgelist = pd.read_csv(filepath, delimiter=", ")
    G = nx.from_pandas_edgelist(edgelist, source = 'Character1', target = 'Character2', edge_attr=True)

    # create a subgraph with the specified nodes
    if nodelist != None:
        G = nx.subgraph(G, nodelist)
    return G






def SSBM(N, degree, noise, n_factions=2, faction_sizes='equal'):

    # generates a signed stochastic block model
    # input:
    # N: number of nodes
    # degree: mean degree
    # noise: probability of an edge being flipped
    # n_factions: number of communities or factions in the graph
    # faction_sizes = {'equal', 'random'}: Sizes of the factions. If 'equal', all groups have the same size. If 'random', the sizes are random, each faction has a random size
    # output:
    # G: a networkx graph object
    
    # generate an unsigned Erdos-Renyi graph
    p = degree/N
    G = nx.erdos_renyi_graph(N,p)

    # assign a faction to each node
    if faction_sizes == 'equal':
        factions = np.zeros(N, dtype = int)
        for i,node in enumerate(G.nodes()):
            factions[i] = i%n_factions
            G.nodes[node]['SSBM_faction'] = i%n_factions

    elif faction_sizes == 'random':
        
        def determine_faction_sizes(N, n_factions):
     
            faction_sizes = np.random.rand(n_factions)  # generate random sizes for the factions
            faction_sizes = np.floor(faction_sizes/sum(faction_sizes)*N).astype(int)  # normalize the sizes so that they sum to N
            while sum(faction_sizes) != N:  # in case the sum of the sizes is not N, add nodes to random factions
                faction_sizes[np.random.randint(n_factions)] += 1
                return faction_sizes

        faction_sizes = determine_faction_sizes(N, n_factions)
        while (faction_sizes == 0).any() or (faction_sizes == 1).any():  # avoid factions with no node only one node
            faction_sizes = determine_faction_sizes(N, n_factions)

        factions = np.repeat(np.arange(len(faction_sizes)), faction_sizes)  # create the list of factions from the sizes
        for i,node in enumerate(G.nodes()):
            G.nodes[node]['SSBM_faction'] = factions[i]

    
    # assign signs to the edges
    
    for (node1,node2, w) in G.edges(data='weight'):

        if factions[node1] == factions[node2]: # same community
            u = np.random.rand()
            if u < 1-noise:
                G[node1][node2]['weight'] = 1
            else:
                G[node1][node2]['weight'] = -1

        else: # different communities
            u = np.random.rand()
            if u < noise:
                G[node1][node2]['weight'] = 1
            else:
                G[node1][node2]['weight'] = -1

    return G





def topological_distance_matrix(A):
    # computes the topological distance matrix from a given adjacency matrix
    # input: A: adjacency matrix (weighted)
    # output: distance: shortest-path distance matrix 

    G = nx.from_numpy_array(A)
    dist_dict = dict(nx.shortest_path_length(G, weight = 'weight'))   # compute topological distance
    
    distance = np.zeros((len(G.nodes()),len(G.nodes())))    # initialize distance matrix
    for i in G.nodes():
        for j in G.nodes():
            if j not in dist_dict[i].keys():  #if nodes are not connected, assign as infinite distance
                distance[i][j] = np.Inf
            else:
                distance[i][j] = dist_dict[i][j]

    return distance



##############################################################################################################
################ FUNCTIONS TO EVALUATE THE PERFORMANCE OF THE CLUSTERING ALGORITHMS ##########################
##############################################################################################################


def signed_modularity(G, labels):
    from networkx.algorithms.community import modularity
    
    # adapt the labels form the sklearn format to the one taken by the modularity function of networkx
    if isinstance(labels, np.ndarray):
        unique_labels = set(labels)
        mod_labels = [None]*len(unique_labels)
        for i, label in enumerate(unique_labels):
            mod_labels[i] = set(np.where(labels == label)[0])
    elif isinstance(labels, list):
        mod_labels = labels
    else:
        raise ValueError("labels must be a list or a numpy array")

    #create the adjacency matrix A, matrix of positive edges P and matrix of negative edges N
    A = nx.adjacency_matrix(G).toarray()
    P = nx.from_numpy_matrix(A*(A>0) )
    N = nx.from_numpy_matrix(-A*(A<0) )
            
    # calculate the total strength of positive and negative edges
    wp = P.number_of_edges()
    wn = N.number_of_edges()
    
    # calculate the modularity of each subgraph and add them up
    print(mod_labels)
    Qp = modularity(P, mod_labels)
    Qn = modularity(N, mod_labels)
    Q = (wp*Qp - wn*Qn)/(wp+wn)
        
    return Q





def frustration_index(G, labels, normalized = True):
    
    # computes the frustration index of a signed network
    # input:
    # G: networkx Graph object containing the signed network
    # labels: list indicating the group to which each node belongs
    # output:
    # F: frustration index of the network

    N = G.order()
    n_factions = max(labels)+1

    #create the adjacency matrix A, block-partition matrix B, matrix of positive edges P and matrix of negative edges N
    A = nx.adjacency_matrix(G).toarray()
    if n_factions == 1:
        B = np.ones((N,n_factions))
    else:
        B = np.zeros((N,n_factions))
        for i in range(N):
                B[i,labels[i]] = 1
            
    Pos = A*(A>0) 
    Neg = -A*(A<0)
    
    #compute the frustration index
    v1 = np.ones(N)
    numerator = v1.T @ ((B@B.T) *Neg + ((np.ones((N,N))-(B@B.T)) *Pos) ) @ v1    
    denominator = v1.T@np.abs(A)@v1
    F = (numerator/denominator) if normalized == True else numerator

    
    return F


