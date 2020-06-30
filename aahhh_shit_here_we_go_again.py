import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
 
 
def plot_weighted_graph(node_list,pos):
    
    #Graph obj 
    #TODO : zufällig nodelist
    # kanten setzen
    # länge bestimmen 
    G = nx.Graph() 
    
    #random nodelist
    
    copy=node_list.copy()
    depot=copy[:1]
    copy=copy[1:]
    while True:
        leave=False
        np.random.shuffle(copy)
        copy.insert(len(copy),depot[0])
        for x in (0,len(node_list)-1):
            if copy[x]==node_list[x]:
                break
            if x<len(node_list)-1:
                node_list[x+1]=copy[x]
            if x==len(copy)-1:
                leave=True
        if leave:
            break

    print(node_list)
    print(copy)



    for node,node2 in zip(node_list,copy):
        G.add_node(node,pos=pos[node])
        G.add_edges_from([(node,node2)])

    #G.add_weighted_edges_from([('Depot','A',12),('Depot','B',13)])
    #elarge = [(d) for (u, v, d) in G.edges(data=True)]
    #print(elarge)
    nx.draw_networkx_nodes(G,pos)
    nx.draw_networkx_labels(G,pos,font_size=10, font_family="sans-serif")
    nx.draw_networkx_edges(G,pos)
    
    plt.draw()
    plt.show()
    #labels = {}
    # for node_name in node_list:
    #     labels[str(node_name)] =str(node_name)
    # nx.draw_networkx_labels(G,pos,labels,font_size=16)


node_list = ['Depot','A','B','C','D','E','F']
pos={'Depot':(0,0),'A':(220,20),'B':(270,70),'C':(250,210),'D':(90,100),'E':(120,110),'F':(50,220)}

plot_weighted_graph(node_list,pos)