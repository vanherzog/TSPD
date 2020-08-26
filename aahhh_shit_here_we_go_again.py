import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
 
 
def plot_weighted_graph(node_list,pos):
    

    G = nx.Graph() 
    
    #random nodelist
    
    copy=node_list.copy()
    depot=copy[:1]
    copy=copy[1:]

    np.random.shuffle(copy)
    copy.insert(len(copy),depot[0])
    for x in range (1,len(node_list)):
        node_list[x]=copy[x-1]

    #add nodes and edges in graph
    for node,node2 in zip(node_list,copy):
        G.add_node(node,pos=pos[node])
        G.add_edges_from([(node,node2)])

    #edge range
    pos1=[]
    pos2=[]
    weite=[]
    for counter in range(len(copy)):
        pos1.insert(counter,pos[node_list[counter]])
        pos2.insert(counter,pos[copy[counter]])
        weite.insert(counter,math.hypot(round(abs(  pos1[counter][0]-pos2[counter][0]  ),1),round(abs( pos1[counter][1]-pos2[counter][1] ),1)))
    for counter in range (len(weite)):
        weite[counter]=round(weite[counter],1)
        nx.draw_networkx_edge_labels(G,pos,edge_labels={(node_list[counter],copy[counter]):weite[counter]})

    #draws
    nx.draw_networkx_nodes(G,pos)
    nx.draw_networkx_labels(G,pos,font_size=10, font_family="sans-serif")
    nx.draw_networkx_edges(G,pos)
    nx.draw_networkx_edge_labels(G,pos,edge_labels={(node_list[counter],copy[counter]):weite[counter]})

    
    plt.draw()
    plt.show()
    print(node_list)
    print(weite)

    return weite,node_list

node_list = ['Depot','A','B','C','D','E','F'] #vlt besser einfach die Anzahl an Punkten übergeben
pos={'Depot':(0,0),'A':(220,20),'B':(270,70),'C':(250,210),'D':(90,60),'E':(120,120),'F':(50,220)}

values,node_list=plot_weighted_graph(node_list,pos)


