import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
 


    
N=2
G=nx.grid_2d_graph(N,N)
inds=labels.keys()
vals=labels.values()
plt.figure()
nx.draw_networkx(G, pos=pos2, with_labels=True, node_size = 200)

plt.draw()
plt.show()

#pos = dict( (n, n) for n in G.nodes() )
#labels = dict( [(1,'111'), (2,'222'), (3,'333'),(4,'444')])
#nx.relabel_nodes(G,labels,False)
#inds=labels.keys()
#vals=labels.values()
#pos2=dict(zip(vals,inds))