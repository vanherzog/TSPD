import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import pandas
 
 
def plot_weighted_graph(node_list,pos):
    
    G = nx.MultiDiGraph(format='png', directed=True)
    
    
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
    # plt.draw()
    # plt.show()
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
    #Kleinste Knoten Matrix, mit den jeweiligen nähesten j zu i und k
    jKnotenM = np.zeros(shape=(len(numbers),len(numbers)))
    #initialisierung
    kk = None

    #none setzen array
    #Hilfsabstandsmatrix
    ha = np.zeros(shape=(len(numbers),len(numbers)))
    #Kostenmatrix
    kostenM = np.zeros(shape=(len(numbers),len(numbers)))
    print(len(ha[0]))
    print(len(ha))
    for i in range (len(numbers)):
        for j in range (len(numbers)):
            ha[i][j] = None
            kostenM[i][j] = None
            jKnotenM[i][j] = None

    
    #kostenM = ha

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
            c = 400
            ct = 399
            for j in range(i+1, len(numbers)):
                if(numbers[j] == numbers[k]):
                    break
                #addieren beider drohnenrouten
                dt = ab[numbers[i]][numbers[j]] + ab[numbers[j]][numbers[k]] 
                ct = kosten(numbers[i], numbers[j], numbers[k], 8.3, 5.5, 8, 20)
                if(ct<c):
                    c = ct
                    d = dt

                    #kk = kleinster Knoten
                    kk = numbers[j]      
            jKnotenM[numbers[i]][numbers[k]]= kk
            kk = None

            #300 ist die weiteste entfernung die die Drohne fliegen könnte (Limit)
            if(c<300):
                ha[numbers[i]][numbers[k]]= round(d,2) 
                kostenM[numbers[i]][numbers[k]] = "{:f}".format(float(c))
            else:
                ha[numbers[i]][numbers[k]]= 9999 #unendlich
                kostenM[numbers[i]][numbers[k]]= 9999 #unendlich

    return ha,jKnotenM,kostenM

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
    
    for a in range (len(numbers)):
        for b in range (len(numbers)):
            ab[a][b]= round(math.hypot( round(abs( xcoord[a]-xcoord[b]  ),1) ,round(abs(  ycoord[a]-ycoord[b] ),1) ),1)
    return ab

def test(ha):
    edges=[]
    for i in range (len(ha)):
        for j in range (len(ha)):
            if ha[i][j]!=None:
                edges.append((str(i),str(j)))
    print(edges)

def print_hilfsgraph(ha):
    # Graph data

    #zB [0, 1, 2, 3, 6, 4, 5, 7]
    names = []
    for number in numbers:
        names.append(str(number))
    positions = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0)]
    edges = []

    
    # Matplotlib figure
    plt.figure('Hilfsgraph')

    # Create graph
    G = nx.MultiDiGraph(format='png', directed=True)

    for index, name in enumerate(names):
        G.add_node(name, pos=positions[index])



    #ha durchlaufen, kanten hinzufügen, beschriftungen hinzufügen
    for i in range (len(ha)):
        for j in range (len(ha)):
            if ha[i][j]>0:
                edges.append((str(j),str(i)))
                #nx.draw_networkx_edge_labels(G,positions,edge_labels={(i,j):ha[i][j]})


    
    layout = dict((n, G.nodes[n]["pos"]) for n in G.nodes())
    nx.draw(G, pos=layout, with_labels=True, node_size=300)
    ax = plt.gca()
    for edge in edges:
        ax.annotate("",
                    xy=layout[edge[0]], xycoords='data',
                    xytext=layout[edge[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=-0.7",
                                    ),
                    )
    
    nx.draw_networkx_labels(G,posnumbers,font_size=1, font_family="sans-serif")
    plt.draw()
    plt.show()


def kosten(i, j, k, gD, gT, wD, wT):

    warteKostenT = 0
    warteKostenD = 0

    #Zeit vom Truck 
    zeitSubIK= ab[i][k] / gT

    #Zeit der Drohne
    zeitIJK = (ab[i][j] + ab[j][k]) /gD

    #Einsparung der geänderten Strecke des Truck
    einsparung = (ab[i][k] - ab[i][j] - ab[j][k]) /gT 

    #Wer muss warten
    dif = zeitIJK - zeitSubIK

    #WarteKosten des Truck und der Drohne
    if(dif > 0):
        warteKostenT = dif * wT
    else: 
        warteKostenD = dif * wD

    #die Drohne ist 25mal billiger als der Truck
    costIJK = zeitIJK/25
    costSubIK = zeitSubIK  

    #nach der Formel in 44
    cost = costSubIK + costIJK + einsparung + warteKostenD + warteKostenT

    return cost








node_list = ['X','A','B','C','D','E','F'] 
pos={'X':(0,0),'A':(220,20),'B':(270,70),'C':(250,210),'D':(90,60),'E':(120,120),'F':(50,220)}
posnumbers={'0':(0,0),'1':(220,20),'2':(270,70),'3':(250,210),'4':(90,60),'5':(120,120),'6':(50,220), '7':(0,0)}


row_labels = ['X','A', 'B', 'C', 'D', 'E', 'F', 'X']
column_labels = ['X','A', 'B', 'C', 'D', 'E', 'F', 'X']

weite,node_list=plot_weighted_graph(node_list,pos)
numbers=umschreiben(node_list)
ab= abstaende(numbers)
ha,jKnotenM,kostenM = hilfsgraph(weite,numbers,ab)
print('alle Abstände')
print(ab)
print('ha')
print(ha)
print('Kosten')
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})
dfkostenM = pandas.DataFrame(kostenM, columns=column_labels, index=row_labels)

myArray=np.array(dfkostenM)
print(myArray)
print('Knotenmatrix')
print (jKnotenM)
#print_hilfsgraph(ha)



