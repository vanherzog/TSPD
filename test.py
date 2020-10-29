import networkx as nx
import matplotlib.pyplot as plt
import random
n = random.randint(0,22)
print(n)

J = nx.MultiDiGraph(format='png', directed=True)
G = nx.MultiDiGraph(format='png', directed=True)

def erste_methode(pos):
    global G
    
    for node,node2 in zip(node_list[1:],node_list[:len(node_list)-1]):
        G.add_node(node,pos=pos[node])
        G.add_edges_from([(node,node2)])
    nx.draw_networkx_nodes(G,pos)
    nx.draw_networkx_edges(G,pos,edge_color="b")
    nx.draw_networkx_labels(G,pos,font_size=10, font_family="sans-serif")

def test_methode():
    global J
    #J = G.__class__()
    J.add_nodes_from(G)
    
pos={'X':(0,0),'A':(220,20),'B':(270,70),'C':(250,210),'D':(90,60),'E':(120,120),'F':(50,220)}
node_list = ['X','A','B','C','D','E','F']    


print('hallo`')
erste_methode(pos)
f1=plt.figure('drölf')
plt.show