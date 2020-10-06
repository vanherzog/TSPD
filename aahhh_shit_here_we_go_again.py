import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import pandas as pd
 
G = nx.MultiDiGraph(format='png', directed=True)

def plot_weighted_graph(node_list,pos):
    
    
    
    
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
            d = 399
            dt = 400
            c = 400
            ct = 399
            for j in range(i+1, len(numbers)):
                if(numbers[j] == numbers[k]):
                    break

                #addieren beider drohnenrouten (wo keine drohnenrouten gemacht werden, also bei direkten nachbarn wird die normale distanz genommen)
                dt = ab[numbers[i]][numbers[j]] + ab[numbers[j]][numbers[k]] 
                ct = kosten(numbers[i], numbers[j], numbers[k],gD, gT, wD, wT)
                
                if(ct<c):
                    c = ct
                    d = dt

                    #kk = kleinster Knoten
                    kk = numbers[j]      
                    jKnotenM[numbers[i]][numbers[k]]= kk
                    #kk = None

            #400 ist die weiteste entfernung die die Drohne fliegen könnte (Limit)
            if(c<400):
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

H = nx.MultiDiGraph(format='png', directed=True)

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
    # plt.draw()
    # plt.show()


def kosten(i, j, k, gD, gT, wD, wT):
    #i j k sind numbers[i,j,k] zb(Knoten 5,3,6)
    warteKostenT = 0
    warteKostenD = 0

    #Zeit vom Truck 
    abSum=0
    
    #kumulieren der Truckstrecke
    for x in range (numbers.index(i),numbers.index(k)-1):
        if numbers.index(j)==x:
            i=i+1
        if x+1==numbers.index(j):
            abSum=abSum+ab[x][x+2]
        else:
            abSum=abSum+ab[x][x+1]

    zeitSubIK= abSum / gT

    #Zeit der Drohne
    zeitIJK = (ab[i][j] + ab[j][k]) /gD

    #Einsparung der geänderten Strecke des Truck
    #einsparung = (ab[i][k] - ab[i][j] - ab[j][k]) /gT 
    

    #Wer muss warten
    dif = zeitIJK - zeitSubIK

    #WarteKosten des Truck und der Drohne
    if(dif > 0):
        warteKostenT = abs(dif) * wT
    else: 
        warteKostenD = abs(dif) * wD

    #die Drohne ist 25mal billiger als der Truck
    costIJK = zeitIJK/25
    costSubIK = zeitSubIK  


    #einsparung positiv
    
    #nach der Formel in 44
    cost = costSubIK + costIJK + warteKostenD + warteKostenT

    return cost

def direkteTruckKosten(ab,gT):
    dkm = np.zeros(shape=(len(numbers),len(numbers)))
    #direkte Abstände in numbers eintragen (weite[i] hat schon kosten drin)
    for i in range(len(numbers)-1):
        dkm[numbers[i]][numbers[i+1]]=weite[i]
    #alle anderen abstände eintragen (bögen im Hilfsgraph) abstände werden kumuliert mit jeder weiteren Pos
    for j in range(len(numbers)-2):
        for k in range (j+1,len(numbers)-1):
            dkm[numbers[j]][numbers[k+1]]=dkm[numbers[j]][numbers[k]]+dkm[numbers[k]][numbers[k+1]]
    return dkm

def findingshortP():
    
    P = [None] * len(numbers)
    V = [9999] * len(numbers)
    
    V[0] = 0
    P[0] = 0

    #dijkstra
    for k in range(1,len(numbers)):
        direkt = dkm[numbers[k]][numbers[k-1]]+V[numbers[k-1]]
        indirekt = 9999
        finalI = 0
        finalJ = 0
        for i in range(k):
            tmpJ = jKnotenM[numbers[i]][numbers[k]]
            if math.isnan(tmpJ):
                continue
            print(tmpJ)
            tmpJcost = V[numbers[i]]+kosten(numbers[i], numbers[int(tmpJ)], numbers[k], gD, gT, wD, wT)
            if tmpJcost < indirekt:
                indirekt = tmpJcost
                finalJ = tmpJ
                finalI = i
        # if indirekt > direkt:
        #     mydict[i][k]='T'


    return P,V




def billigste(i,k,app):
    x = 'T'
    
    if(dkm[numbers[i]][numbers[k]] <= kostenM[numbers[i]][numbers[k]]):
        kosten = dkm[numbers[i]][numbers[k]]
    else:
        #doppel append weil die drohne zuerst zu j und dann zu k fliegt
        x = 'D'
        kosten = kostenM[numbers[i]][numbers[k]]
    #wenn V[k] noch unendlich ist weil er es gerade erhöht hat, soll der nicht alle ausgerechneten Sachen bis dahin überschreibern, sondern vom letzen Punkt k-1
    #zum neuen k einfach eine Truckroute machen
    # if app == 1 and i == 0:
    #     mitWas[numbers[k-1]][numbers[k]] = 'T'

    # #Truckbedingungen
    # if app == 1 and i != 0 and x == 'T':

    #     #dürfen keine Nachbarn sein, sonst schmeißt index ein Error weil Nachbarn keinen j Knoten haben
    #     if i+1 != k:
    #         #das alte 'T' das dem jetzigen Drohnenpunkt zugeordnet ist, löschen
    #         index = numbers.index(int(jKnotenM[numbers[i]][numbers[k]]))
    #         if mitWas[numbers[index-1]][numbers[index]] == 'T':
    #             mitWas[numbers[index-1]][numbers[index]] = '/'
    #         altesI = 0
    #         altesK = 0
    #         for o in range(i,k-1):
    #             for p in range(i+1,k):
    #                 #alle alten drohnenrouten überschreiben
    #                 if mitWas[numbers[o]][numbers[p]] == '1':
    #                     altesI = numbers[o]
    #                     #keine T's wo ne drohne fliegt(j)
    #                     if o == index-1 and p == index or o == index and p == index+1:
    #                         break   
    #                     if o+1 == p:
    #                         mitWas[numbers[o]][numbers[p]] = 'T'
    #                     else:
    #                         mitWas[numbers[o]][numbers[p]] = '/'

    #                 if mitWas[numbers[o]][numbers[p]] == '2':
    #                     altesK = numbers[p]
    #                     mitWas[altesI][altesK] = '/'
    #                     #keine T's wo ne drohne fliegt(j)
    #                     if o == index-1 and p == index or o == index and p == index+1:
    #                         break   
    #                     if o+1 == p:
    #                         mitWas[numbers[o]][numbers[p]] = 'T'
    #                     else:
    #                         mitWas[numbers[o]][numbers[p]] = '/'
        
        
    #     for n in range(i,k):
    #         mitWas[numbers[n]][numbers[n+1]]= 'T'

    # #Drohnenbedingungen
    # if app == 1 and x == 'D':
    #     alteI = 0
    #     alteK = 0
    #     #das alte 'T' das dem jetzigen Drohnenpunkt zugeordnet ist, löschen
    #     index = numbers.index(int(jKnotenM[numbers[i]][numbers[k]]))
    #     if mitWas[numbers[index-1]][numbers[index]] == 'T':
    #         mitWas[numbers[index-1]][numbers[index]] = '/'
    #     for o in range(i,k-1):
    #         for p in range(i+1,k):
    #             #alle alten drohnenrouten überschreiben, die innerhalb der neuen liegen
    #             if mitWas[numbers[o]][numbers[p]] == '1':
    #                 alteI = numbers[o]
    #                 #keine T's wo ne drohne fliegt(j)
    #                 if o == index-1 and p == index or o == index and p == index+1:
    #                     break   
    #                 #wenn 1 zwischen zwei Nachbarn ist, wird es durch T ersetzt, sonst wird es gelöscht
    #                 if o+1 == p:
    #                     mitWas[numbers[o]][numbers[p]] = 'T'
    #                 else:
    #                     mitWas[numbers[o]][numbers[p]] = '/'

    #             if mitWas[numbers[o]][numbers[p]] == '2':
    #                 alteK = numbers[p]
    #                 mitWas[alteI][alteK] = '/'
    #                 #keine T's wo ne drohne fliegt(j)
    #                 if o == index-1 and p == index or o == index and p == index+1:
    #                     break   
    #                 if o+1 == p:
    #                     mitWas[numbers[o]][numbers[p]] = 'T'
    #                 else:
    #                     mitWas[numbers[o]][numbers[p]] = '/'
    #     #alle Drohnenrouten deren k in der neuen drohnenroute liegt(können teils außerhalb sein), müssen auch gelöscht werden, weil es nur eine Drohne gibt
    #     if i != 0:
    #         for e in range(0,numbers[i-1]):
    #             for f in range(numbers[i+1], numbers[k]):
    #                 try:
    #                     #Nachbarn haben kein j und sonst wird ein ERROR geschmissen
    #                     if jKnotenM[numbers[e]][numbers[f]] >= 0 :
    #                         jj = numbers.index(int(jKnotenM[numbers[e]][numbers[f]]))
    #                         if mitWas[numbers[e]][numbers[jj]] == '1':
    #                             mitWas[numbers[e]][numbers[jj]] == '/'
    #                         if mitWas[numbers[jj]][numbers[f]] == '2':
    #                             mitWas[numbers[jj]][numbers[f]] == '/'
    #                 except ValueError:
    #                     print('############################')
    #                     print (jKnotenM[numbers[e]][numbers[f]])
                                   
            
        
    #     mitWas[numbers[i]][int(jKnotenM[numbers[i]][numbers[k]])]= '1' 
    #     mitWas[int(jKnotenM[numbers[i]][numbers[k]])][numbers[k]]= '2'
    #     #die neue Truckroute eintragen, die den Drohnenpunkt überspringt
    #     mitWas[numbers[index-1]][numbers[index+1]] = 'T'
        
   
    return kosten


# def selectMarix(x):
#     return {
#         '1':M_A,
#         '2':M_B,
#         '3':M_C,
#         '4':M_D,
#         '5':M_E,
#         '6':M_F,
#         '7':M_X,
#     }.get(x,None)


# def Split_Algo_Step2():

#     j = len(numbers) -1
#     i = 9999
#     Sa = []
#     Sa.append(numbers[j])
    
#     while i != 0 :
#         i = P[j]
#         Sa.append(i)
#         j = i

#     #reverse    
#     Sa = Sa[::-1]

    



#     return Sa



node_list = ['X','A','B','C','D','E','F'] 
pos={'X':(0,0),'A':(220,20),'B':(270,70),'C':(250,210),'D':(90,60),'E':(120,120),'F':(50,220)}
posnumbers={'0':(0,0),'1':(220,20),'2':(270,70),'3':(250,210),'4':(90,60),'5':(120,120),'6':(50,220), '7':(0,0)}



gD=9
gT=7
wD=9
wT=12

row_labels = ['X','A', 'B', 'C', 'D', 'E', 'F', 'X']
column_labels = ['X','A', 'B', 'C', 'D', 'E', 'F', 'X']
#Legende
#ha= Hilfsmatrix entspricht hilfsgraphen mit drohnenrouten
#ab= Alle abstände zwischen allen Knoten
#kostenM= hilfsgraph mit Kostenfunktion
#jKnotenM= Knotenmatrix (alle Knoten(j) die für position (i,k) verwendet werden)
#direkteKostenmatrix= ab/truckkosten für direkte nachbarn
#billigstekostenMatrix= ha/kosten für alle drohnenfahrten
weite,node_list=plot_weighted_graph(node_list,pos)
numbers=umschreiben(node_list)
ab= abstaende(numbers)
ha,jKnotenM,kostenM = hilfsgraph(weite,numbers,ab)
dkm=direkteTruckKosten(ab,gT)

mitWas = np.empty(shape=(len(numbers),len(numbers)), dtype = str)
for i in range (len(numbers)):
        for j in range (len(numbers)):
            mitWas[i][j] = '/'

P,V = findingshortP()



M_X= [[None for x in range(len(numbers))] for y in range(len(numbers))] 
M_A= [[None for x in range(len(numbers))] for y in range(len(numbers))] 
M_B= [[None for x in range(len(numbers))] for y in range(len(numbers))] 
M_C= [[None for x in range(len(numbers))] for y in range(len(numbers))] 
M_D= [[None for x in range(len(numbers))] for y in range(len(numbers))] 
M_E= [[None for x in range(len(numbers))] for y in range(len(numbers))] 
M_F= [[None for x in range(len(numbers))] for y in range(len(numbers))] 
# mydict["1"]=M_X
# print(mydict)
dfM_X = pd.DataFrame(M_X, columns=column_labels, index=row_labels)
dfM_A = pd.DataFrame(M_A, columns=column_labels, index=row_labels)
dfM_B = pd.DataFrame(M_B, columns=column_labels, index=row_labels)
dfM_C = pd.DataFrame(M_C, columns=column_labels, index=row_labels)
dfM_D = pd.DataFrame(M_D, columns=column_labels, index=row_labels)
dfM_E = pd.DataFrame(M_E, columns=column_labels, index=row_labels)
dfM_F = pd.DataFrame(M_F, columns=column_labels, index=row_labels)

    
print('M_X',dfM_X, sep='\n')
print('M_A',dfM_A, sep='\n')
print('M_B',dfM_B, sep='\n')
print('M_C',dfM_C, sep='\n')
print('M_D',dfM_D, sep='\n')
print('M_E',dfM_E, sep='\n')
print('M_F',dfM_F, sep='\n')
#pandas
pd.options.display.float_format = '{:0.0f}'.format
print('')

dfdkm = pd.DataFrame(dkm, columns=column_labels, index=row_labels)
print('dkm',dfdkm, sep='\n')

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

mitW = pd.DataFrame(mitWas, columns=column_labels, index=row_labels)
print('mit Was',mitW, sep='\n')

print('')

print('')
print(numbers)
print('')
dfP = pd.DataFrame(P, columns=['X'], index=row_labels)
print('P',dfP.T, sep='\n')
print('')
dfV = pd.DataFrame(V, columns=['X'], index=row_labels)
print('V',dfV.T, sep='\n')
print('')
# Sa = Split_Algo_Step2()
# print ('Sa')
# print(Sa)



plt.draw()
plt.show()