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

    weite.insert(len(weite),None)
    plt.draw()
    plt.show()
    print(node_list)
    print(weite)

    

    return weite,node_list


def umschreiben(node_list):

    numbers = []
    for letter in node_list:
        number = ord(letter) - 64
        numbers.append(number)
    x = numbers.index(24)
    numbers[x] = 0

    # numbers ums depot erweitern
    numbers.insert(len(numbers),7)
    print(numbers)

    print(numbers)
    return numbers


def hilfsgraph(weite, numbers, ab):
    
    #none setzen array
    ha = np.zeros(shape=(len(numbers),len(numbers)))

    for i in range (len(numbers)):
        for j in range (len(numbers)):
            ha[i][j] = None

    #befüllen der Grade
    for i in range (len(numbers)-1):
        ha[numbers[i]][numbers[i+1]]=weite[i]
    ha[numbers[-1]][numbers[0]] = weite[-1]

    
    
    #minIndex = position von j speichern

    for i in range(len(numbers)-2):
        for k in range(i+2, len(numbers)):
            #d ist der kürzeste weg von allen j verschiebungen, mit festem i und k
            #dt ist der weg vom temporären j
            d = 399
            dt = 400
            for j in range(i+1, len(numbers)):
                if(numbers[j] == numbers[k]):
                    break
                #addieren beider dronenrouten
                dt = ab[numbers[i]][numbers[j]] + ab[numbers[j]][numbers[k]] # +Kostenberechnung
                if(dt<d):
                    d = dt
            #1000 ist die weiteste entfernung die die Drohne fliegen könnte (Limit)
            if(d<300):
                ha[numbers[i]][numbers[k]]= d
            else:
                ha[numbers[i]][numbers[k]]= 9999 #unendlich
    print('ha = ')
    print(ha)
    return ha

#abstandsmatrix alleknoten
def abstaende(numbers):
    ab = np.zeros(shape=(len(numbers),len(numbers)))

    for i in range (len(numbers)):
        for j in range (len(numbers)):
            ab[i][j] = None

    xcoord=[]
    ycoord=[]
    for x in range (len(numbers)):
        xcoord.insert(x,posnumbers[str(x)][0])
        ycoord.insert(x,posnumbers[str(x)][1])

    print('coord')
    print(xcoord)
    print(ycoord)
    
    for a in range (len(numbers)):
        for b in range (len(numbers)):
            ab[a][b]= round(math.hypot( round(abs( xcoord[a]-xcoord[b]  ),1) ,round(abs(  ycoord[a]-ycoord[b] ),1) ),1)
    return ab
    

node_list = ['X','A','B','C','D','E','F'] 
pos={'X':(0,0),'A':(220,20),'B':(270,70),'C':(250,210),'D':(90,60),'E':(120,120),'F':(50,220)}
posnumbers={'0':(0,0),'1':(220,20),'2':(270,70),'3':(250,210),'4':(90,60),'5':(120,120),'6':(50,220), '7':(0,0)}

weite,node_list=plot_weighted_graph(node_list,pos)
numbers=umschreiben(node_list)
ab= abstaende(numbers)
ha = hilfsgraph(weite,numbers,ab)
print(ab)



