import numpy as np
import networkx as nx
import matplotlib.pyplot as plt




def draw_network(G, ax = None,  pos = None, labels = None, node_size = 500, node_color = 'white', nodeedge_color = 'k', cmap_nodes = None, label_fontsize = 12, pos_edge_width = 2, neg_edge_width = 2, with_labels = False):
        
    ### draw signed network clustering nodes of the same party together ###
     
    if ax == None:
        fig, ax = plt.subplots(figsize=(10,10))
        
    # if position is unspecified, use kamada_kawai layout of the unsigned network
    if pos == None: 
        G_abs = nx.Graph()
        for u, v, w in G.edges(data = 'weight'):
            G_abs.add_edge(u,v)
        pos = nx.kamada_kawai_layout(G_abs)
        
    # check that all edges have a weight. If they don't have it, assign them a weight 1
    for u, v, w in G.edges(data = 'weight'):
        if w == None:
            G[u][v]['weight'] = 1

    # set edge attributes
    edge_color = ['']*G.number_of_edges()
    ls = ['']*G.number_of_edges()
    width = np.zeros(len(ls))
    for i, (u, v, w) in enumerate(G.edges(data = True)):
        edge_color[i] = 'darkgreen' if w['weight']>0 else 'red'
        ls[i] = '-' if w['weight']>0 else '--'
        width[i] = pos_edge_width if w['weight']>0 else neg_edge_width
    
    # plot the network    
    if cmap_nodes is not None: nodes = nx.draw_networkx_nodes(G,  pos = pos, ax = ax, node_size = node_size, node_color = node_color, cmap = cmap_nodes)
    else: nodes = nx.draw_networkx_nodes(G,  pos = pos, ax = ax, node_size = node_size, node_color = node_color)
    nodes.set_edgecolor(nodeedge_color)
    nx.draw_networkx_edges(G, pos = pos, ax = ax,  edge_color = edge_color, style = ls, width = width)
    if with_labels == True: 
        nx.draw_networkx_labels(G, pos = pos, ax = ax, labels = labels, font_size = label_fontsize)



