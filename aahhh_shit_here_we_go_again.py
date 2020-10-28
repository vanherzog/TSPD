import matplotlib.pyplot as plt
from matplotlib import interactive
import networkx as nx
import numpy as np
import math
import pandas as pd
from copy import deepcopy
 


def plot_weighted_graph(node_list,pos):
    
    #random nodelist
    f1=plt.figure('Random')

    
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

    #befüllen der Grade und einbringen der Geschwindigkeit in die abstände
    
    for i in range (len(numbers)-1):
        ha[numbers[i]][numbers[i+1]]=weite[i]
    ha[numbers[-1]][numbers[0]] = weite[-1]
    
    #weite mit kosten versehen
    for i in range(len(weite)-1):
        weite[i] = round(weite[i]/gT,2)

    
    for i in range (len(numbers)-1):
        kostenM[numbers[i]][numbers[i+1]]=weite[i]
    kostenM[numbers[-1]][numbers[0]] = weite[-1]
    

    
    
    #minIndex = position von j speichern

    for i in range(len(numbers)-2):
        for k in range(i+2, len(numbers)):
            #d ist der kürzeste weg von allen j verschiebungen, mit festem i und k
            #dt ist der weg vom temporären j
            d = 599
            dt =600
            c = 600
            ct = 599
            for j in range(i+1, len(numbers)):
                if(numbers[j] == numbers[k]):
                    break

                #addieren beider drohnenrouten (wo keine drohnenrouten gemacht werden, also bei direkten nachbarn wird die normale distanz genommen)
                dt = ab[numbers[i]][numbers[j]] + ab[numbers[j]][numbers[k]] 
                ct = kosten(numbers[i], numbers[j], numbers[k],gD, gT, wD, wT,numbers)
                
                if(ct<c):
                    c = ct
                    d = dt

                    #kk = kleinster Knoten
                    kk = numbers[j]      
                    jKnotenM[numbers[i]][numbers[k]]= kk
                    #kk = None

            #400 ist die weiteste entfernung die die Drohne fliegen könnte (Limit)
            if(c<600):
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

#k-nearest Neighbour = ändert die Reihenfolge von Numbers und dementsprechend auch weite weil es neue Nachbarn gibt
def knn():
    f4=plt.figure('knn')
    #graph referenzieren
    global I
    I = G.__class__()
    I.add_nodes_from(G)

    OG = ['X','A','B','C','D','E','F']
    #neue Numbers & weite
    new = []
    newWeite = []
    #Depot ist IMMER am Anfang
    new.append(0)
    best = deepcopy(999)
    i = 0
    #Ich appende bei new immer Sachen dazu nach jedem Durchlauf bis es so lang ist wie OG
    while len(new) < len(OG):
        #alle Knoten durchlaufen um den nähesten zu finden
        for j in range (len(OG)):
            #Abstand ausrechnen
            tbest = ab[i][j]
            #Wenn der aktuelle Knoten(tbest) näher ist als der bisher gespeicherte(best) und dieser Knoten noch nicht hinzugefügt wurde
            if tbest < best and j not in new:
                best = deepcopy(tbest)
                save = deepcopy(j)
        #Wenn alle Knoten durchgelaufen wurden, den nähesten hinzufügen und auch die dazugehörige weite 
        new.append(save)
        newWeite.append(best)
        best = deepcopy(999)
        #Jetzt für den neu zugefügten Knoten, wieder den nähesten suchen
        i = save
    #Depot ist IMMER das Ende
    new.append(7)
    #Die letzte Enfernung zum Depot hinzufügen und NONE weil vom ENDDEPOT(X) zum ANFANGSDEPOT(XO) kein Abstand mehr ist
    newWeite.append(ab[save][7])
    newWeite.append(None)
    #graphen konstruieren
    for k in range(len(OG)-1):
        I.add_edges_from([(OG[new[k]],OG[new[k+1]])])
        nx.draw_networkx_edge_labels(J,pos,edge_labels={(OG[new[k]],OG[new[k+1]]): str(int(round(newWeite[k],0))) })
    #letzte kante
    I.add_edges_from([(OG[new[len(new)-2]],OG[0])])
    nx.draw_networkx_edge_labels(J,pos,edge_labels={(OG[new[len(new)-2]],OG[0]): str(int(round(newWeite[len(newWeite)-2],0))) })
    
    nx.draw_networkx_nodes(I,pos)
    nx.draw_networkx_labels(I,pos,font_size=10, font_family="sans-serif")
    nx.draw_networkx_edges(I,pos)
                  
    return new, newWeite

#k-cheapest insertion
def kci(numbers):
    f5=plt.figure('kci')
    #graph referenzieren
    global K
    K = G.__class__()
    K.add_nodes_from(G)

    OG = ['X','A','B','C','D','E','F']

    new=[]
    newWeite=[]
    newWeiteInnen=[]
    newWeiteAussen=[]
    copy= deepcopy(numbers)
    subtour=copy[0:4]
    insertions=copy[4:7]
    for i in range(3):
        #nächster subtour Knoten einfügen
        new.append(subtour[i])
        #bester Knoten
        bestKnoten=9
        #beste subtour Weite
        bestWeite=9999
        #insertion Kanten
        bestWeite1=999
        bestWeite2=999
        for j in range(3):
            #weite von d(i,v)+d(v,j)
            weite1 = ab[ subtour[i] ][ insertions[j] ]
            weite2 = ab[ insertions[j] ][ subtour[i+1] ]
            weite = weite1 + weite2
            if weite < bestWeite and insertions[j] not in new:
                bestWeite = weite
                bestKnoten = insertions[j]
                bestWeite1=weite1
                bestWeite2=weite2
        #beste insertion einfügen
        newWeiteInnen.append(round(bestWeite,1))
        newWeiteAussen.append(round(bestWeite1,1))
        newWeiteAussen.append(round(bestWeite2,1))
        new.append(bestKnoten)
        
    
    #letzter aus subtour hinzufügen
    new.append( subtour[len(subtour)-1] )
    #depot hinzufügen
    new.append( numbers[len(numbers)-1] )
    #vom letzten zum depot abstand
    newWeiteInnen.append(round( ab[ new[len(new)-2] ][ new[len(new)-1] ] ,1))
    #newWeite(return) bestimmen
    newWeite=deepcopy(newWeiteAussen)
    newWeite.append(round( ab[ new[len(new)-2] ][ new[len(new)-1] ] ,1))
    newWeite.append(None)

    print(new)
    print(newWeite)
    print(newWeiteInnen)
    print(newWeiteAussen)
    #graphen konstruieren
  
    #äußerer Graph
    for k in range(len(newWeite)-2):
        K.add_edges_from([(OG[new[k]],OG[new[k+1]])])
        nx.draw_networkx_edge_labels(K,pos,edge_labels={(OG[new[k]],OG[new[k+1]]): str(int(round(newWeite[k],0))) })
    K.add_edges_from([(OG[new[len(new)-2]],OG[new[0]])])
    nx.draw_networkx_edges(K,pos,edge_color="b",style='dashdot')
    G.clear()
    #innerer Graph
    for k in range(len(newWeiteInnen)-1):
        K.add_edges_from([(OG[new[k]],OG[new[k+2]])])   
        nx.draw_networkx_edge_labels(K,pos,edge_labels={(OG[new[k]],OG[new[k+2]]): str(int(round(newWeiteInnen[k],0))) })
        k+=1
    nx.draw_networkx_edges(K,pos,edge_color="g",style='dotted')  

    nx.draw_networkx_nodes(K,pos)
    nx.draw_networkx_labels(K,pos,font_size=10, font_family="sans-serif")


    return new, newWeite


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
    #plt.figure('Hilfsgraph')

    # Create graph
    

    for index, name in enumerate(names):
        H.add_node(name, pos=positions[index])



    #ha durchlaufen, kanten hinzufügen, beschriftungen hinzufügen
    for i in range (len(ha)):
        for j in range (len(ha)):
            if ha[i][j]>0:
                edges.append((str(j),str(i)))
                #nx.draw_networkx_edge_labels(G,positions,edge_labels={(i,j):ha[i][j]})


    
    layout = dict((n, H.nodes[n]["pos"]) for n in H.nodes())
    nx.draw(H, pos=layout, with_labels=True, node_size=300)
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
    
    nx.draw_networkx_labels(H,posnumbers,font_size=1, font_family="sans-serif")


def kosten(i, j, k, gD, gT, wD, wT,numbers):
    #i j k sind numbers[i,j,k] zb(Knoten 5,3,6)
    warteKostenT = 0
    warteKostenD = 0

    #Zeit vom Truck 
    abSum=0
    
    #kumulieren der Truckstrecke
    for x in range (numbers.index(i),numbers.index(k)):
        if numbers.index(j)==x:
            x=x+1
        if x+1==numbers.index(j):
            abSum=abSum+ab[x][x+2]
            if x+2==numbers.index(k):
                break
        else:
            abSum=abSum+ab[x][x+1]

    zeitSubIK= abSum / gT

    #Zeit der Drohne
    zeitIJK = (ab[i][j] + ab[j][k]) /gD

    #Wer muss warten
    dif = (zeitIJK - zeitSubIK) / 5

    #WarteKosten des Truck und der Drohne
    if(dif > 0):
        warteKostenT = abs(dif) * wT
    else: 
        warteKostenD = abs(dif) * wD

    #die Drohne ist 25mal billiger als der Truck
    costIJK = zeitIJK/25
    costSubIK = zeitSubIK  
    cost = costSubIK + costIJK + warteKostenD + warteKostenT

    return cost

def findingshortP(numbers,jKnotenM):
    
    P = [None] * len(numbers)
    V = [9999] * len(numbers)
    
    V[0] = 0
    P[0] = 0

    #dijkstra
    for k in range(1,len(numbers)):
        direkt = ab[numbers[k-1]][numbers[k]]/gT + V[numbers[k-1]]
        indirekt = 9999
        finalI = 0
        finalJ = 0
        for i in range(k):
            tmpJ = jKnotenM[numbers[i]][numbers[k]]
            #wenn es kein J gibt, dann gibt es auch keinen Drohnenflug
            if math.isnan(tmpJ):
                continue
            tmpJcost = V[numbers[i]] + kosten(numbers[i], int(tmpJ), numbers[k], gD, gT, wD, wT,numbers)
            #bester Drohnenflug? bzw bestes J?
            if tmpJcost < indirekt:
                indirekt = tmpJcost
                finalJ = deepcopy(tmpJ)
                finalI = deepcopy(i)    

        if direkt < indirekt:
            P[numbers[k]] = numbers[k-1]
            V[numbers[k]] = direkt
            #kopieren des Vorgängers

            copyMatrix(str(numbers[k]),str(numbers[k-1]))
            #benachbarte Truckstrecke

            setMatrix(str(numbers[k]),numbers[k-1],numbers[k],'T')
            
        else:
            P[numbers[k]]=numbers[finalI]
            V[numbers[k]]=indirekt

            #kopieren bis I
            copyMatrix(str(numbers[k]),str(int(numbers[finalI])))

            #Truckstrecke während Drohnenflug
            for x in range (finalI,k):
                if numbers.index(finalJ) == x:
                    x = x+1
                if x+1 == numbers.index(finalJ):
                    setMatrix(str(numbers[k]),numbers[x],numbers[x+2],'T')
                    if x+2 == k:
                        break
                else:
                    setMatrix(str(numbers[k]),numbers[x],numbers[x+1],'T')
            #Drohnenflug
            setMatrix(str(numbers[k]),int(numbers[finalI]),int(finalJ),'1')
            setMatrix(str(numbers[k]),int(finalJ),numbers[k],'2')

            #finalJ ist schon wegen tmpJ nach numbers sortiert


    return P,V 

def drohnenGraph(H,V):
    #kopieren der Knoten
    H = G.__class__()
    H.add_nodes_from(G)
    
    node_list2 = ['X','A','B','C','D','E','F'] 
    nx.draw_networkx_nodes(H,pos)
    nx.draw_networkx_labels(H,pos,font_size=10, font_family="sans-serif")
    for i in range(len(M_X)-1):
        for j in range(len(M_X[0])):
            #Drohnenpfeil
            if M_X[i][j]=='1' or M_X[i][j]=='2':
                #X gleich X_0 
                if j==len(M_X[0])-1:
                    H.add_edges_from([(node_list2[i],node_list2[0])])
                    nx.draw_networkx_edge_labels(H,pos,edge_labels={(node_list2[i],node_list2[0]): str(int(round(ab[i][0]/gD,0))) +'s' })
                else:
                    H.add_edges_from([(node_list2[i],node_list2[j])])
                    nx.draw_networkx_edge_labels(H,pos,edge_labels={(node_list2[i],node_list2[j]):  str(int(round(ab[i][j]/gD,0))) +'s' })
                nx.draw_networkx_edges(H,pos,edge_color="b",style='dotted')
                
            #Truckpfeil
            if M_X[i][j]=='T':
                #X gleich X_0
                if j==len(M_X[0])-1:
                    liste=[[node_list2[i],node_list2[0]]]
                    nx.draw_networkx_edges(H,pos,edge_color="y",edgelist=liste )
                    nx.draw_networkx_edge_labels(H,pos,edge_labels={(node_list2[i],node_list2[0]):str(int(round(ab[i][0]/gT,0))) +'s' })
                else:
                    liste=[[node_list2[i],node_list2[j]]]
                    nx.draw_networkx_edges(H,pos,edge_color="y",edgelist=liste )
                    nx.draw_networkx_edge_labels(H,pos,edge_labels={(node_list2[i],node_list2[j]): str(int(round(ab[i][j]/gT,0))) +'s' })

    offset =15
    pos_labels = {}
    keys = pos.keys()
    labels={}
    r=0
    for key in keys:
        
        print(key)
        x, y = pos[key]
        pos_labels[key] = (x, y+offset)
        if key == 'X':
            labels[key]= int(round(V[len(V)-1]))
        else:
            labels[key]= int(round(V[r]))
        r=r+1
    nx.draw_networkx_labels(H,pos=pos_labels,labels=labels,fontsize=2)               

def setMatrix(m,y,x,sign):


    switcher = {
        '1':M_A,
        '2':M_B,
        '3':M_C,
        '4':M_D,
        '5':M_E,
        '6':M_F,
        '7':M_X
    }.get(m,None)

    switcher[y][x]=sign


def copyMatrix(kM,copy):
    

    global M_X0
    global M_A
    global M_B
    global M_C
    global M_D
    global M_E
    global M_F
    global M_X

    stitcher = {
        '0':M_X0,
        '1':M_A,
        '2':M_B,
        '3':M_C,
        '4':M_D,
        '5':M_E,
        '6':M_F,
        '7':M_X
    }.get(copy,None)

    if kM == '0':
        M_X0= deepcopy(stitcher)
    elif kM == '1':
        M_A=deepcopy(stitcher)
    elif kM == '2':
        M_B=deepcopy(stitcher)
    elif kM == '3':
        M_C=deepcopy(stitcher)
    elif kM == '4':
        M_D=deepcopy(stitcher)
    elif kM == '5':
        M_E=deepcopy(stitcher)
    elif kM == '6':
        M_F=deepcopy(stitcher)
    elif kM == '7':
        M_X=deepcopy(stitcher)

def null_setzen():
    global M_X0
    global M_A
    global M_B
    global M_C
    global M_D
    global M_E
    global M_F
    global M_X
    M_X0= [['/' for x in range(len(numbers))] for y in range(len(numbers))] 
    M_X= [['/' for x in range(len(numbers))] for y in range(len(numbers))] 
    M_A= [['/' for x in range(len(numbers))] for y in range(len(numbers))] 
    M_B= [['/' for x in range(len(numbers))] for y in range(len(numbers))] 
    M_C= [['/' for x in range(len(numbers))] for y in range(len(numbers))] 
    M_D= [['/' for x in range(len(numbers))] for y in range(len(numbers))] 
    M_E= [['/' for x in range(len(numbers))] for y in range(len(numbers))] 
    M_F= [['/' for x in range(len(numbers))] for y in range(len(numbers))] 


G = nx.MultiDiGraph(format='png', directed=True)
H = nx.MultiDiGraph(format='png', directed=True)
I = nx.MultiDiGraph(format='png', directed=True)
J = nx.MultiDiGraph(format='png', directed=True)
K = nx.MultiDiGraph(format='png', directed=True)
L = nx.MultiDiGraph(format='png', directed=True)


node_list = ['X','A','B','C','D','E','F'] 
pos={'X':(0,0),'A':(220,20),'B':(270,70),'C':(250,210),'D':(90,60),'E':(120,120),'F':(50,220)}
posnumbers={'0':(0,0),'1':(220,20),'2':(270,70),'3':(250,210),'4':(90,60),'5':(120,120),'6':(50,220), '7':(0,0)}

gD=14
gT=7
wD=3
wT=12

row_labels = ['X','A', 'B', 'C', 'D', 'E', 'F', 'X']
column_labels = ['X','A', 'B', 'C', 'D', 'E', 'F', 'X']

M_X0= [] 
M_X= [] 
M_A= [] 
M_B= [] 
M_C= [] 
M_D= [] 
M_E= [] 
M_F= []

#Legende
    #ha= Hilfsmatrix entspricht hilfsgraphen mit drohnenrouten
    #ab= Alle abstände zwischen allen Knoten
    #kostenM= hilfsgraph mit Kostenfunktion
    #jKnotenM= Knotenmatrix (alle Knoten(j) die für position (i,k) verwendet werden)
    #direkteKostenmatrix= ab/truckkosten für direkte nachbarn
    #billigstekostenMatrix= ha/kosten für alle drohnenfahrten

#random insertion
weite,node_list=plot_weighted_graph(node_list,pos) #H befüllen
numbers=umschreiben(node_list)
ab= abstaende(numbers)
ha,jKnotenM,kostenM = hilfsgraph(weite,numbers,ab)
null_setzen()
P,V = findingshortP(numbers,jKnotenM)
f2=plt.figure('random Drohnentour')
drohnenGraph(H,V)


#K-nearest neighbour 
numbers2,weite2 = knn()
ha2,jKnotenM2,kostenM2 = hilfsgraph(weite2,numbers2,ab)
null_setzen()
P2,V2 = findingshortP(numbers2,jKnotenM2)
f3=plt.figure('knn Drohnentour')
drohnenGraph(J,V2)


#K-cheapest insertion
numbers3, weite3 =kci(numbers)
ha3,jKnotenM3,kostenM3 = hilfsgraph(weite3,numbers3,ab)
null_setzen()
P3,V3 = findingshortP(numbers3,jKnotenM3)
f6=plt.figure('kci Drohnentour')
drohnenGraph(L,V3)


#pandas
pd.options.display.float_format = '{:0.0f}'.format
print('')

dfab = pd.DataFrame(ab, columns=column_labels, index=row_labels)
print('alle Abstände',dfab, sep='\n')

print('')

dfha = pd.DataFrame(ha, columns=column_labels, index=row_labels)
print('ha',dfha, sep='\n')

print('')

dfkostenM = pd.DataFrame(kostenM, columns=column_labels, index=row_labels)
print("Kosten" , dfkostenM, sep='\n')

print('')

dfjKnotenM = pd.DataFrame(jKnotenM, columns=column_labels, index=row_labels)
print('Knotenmatrix',dfjKnotenM, sep='\n')

print('')
print(numbers)
print('')
dfP = pd.DataFrame(P, columns=['X'], index=row_labels)
print('P',dfP.T, sep='\n')
print('')
dfV = pd.DataFrame(V, columns=['X'], index=row_labels)
print('V',dfV.T, sep='\n')
print('')


dfM_X0 = pd.DataFrame(M_X0, columns=column_labels, index=row_labels)
dfM_X = pd.DataFrame(M_X, columns=column_labels, index=row_labels)
dfM_A = pd.DataFrame(M_A, columns=column_labels, index=row_labels)
dfM_B = pd.DataFrame(M_B, columns=column_labels, index=row_labels)
dfM_C = pd.DataFrame(M_C, columns=column_labels, index=row_labels)
dfM_D = pd.DataFrame(M_D, columns=column_labels, index=row_labels)
dfM_E = pd.DataFrame(M_E, columns=column_labels, index=row_labels)
dfM_F = pd.DataFrame(M_F, columns=column_labels, index=row_labels)

print('M_X0',dfM_X0, sep='\n')  
print('M_A',dfM_A, sep='\n')
print('M_B',dfM_B, sep='\n')
print('M_C',dfM_C, sep='\n')
print('M_D',dfM_D, sep='\n')
print('M_E',dfM_E, sep='\n')
print('M_F',dfM_F, sep='\n')
print('M_X',dfM_X, sep='\n')


plt.show()