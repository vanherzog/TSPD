# On the min-cost Traveling Salesman Problem with Drone

## 1. Introduction

Companies always tend to look for the most cost-efficient methods to distribute goods across logistic networks (Rizzoli et al.,
2007). Traditionally, trucks have been used to handle these tasks and the corresponding transportation problem is modelled as a
traveling salesman problem (TSP). However, a new distribution method has recently arisen in which small unmanned aerial vehicles
(UAV), also known as drones, are deployed to support parcel delivery. On the one hand, there are four advantages of using a drone for
delivery: (1) it can be operated without a human pilot, (2) it avoids the congestion of traditional road networks by flying over them,
(3) it is faster than trucks, and (4) it has much lower transportation costs per kilometre (Wohlsen, 2014). On the other hand, because
the drones are powered by batteries, their flight distance and lifting power are limited, meaning they are restricted in both maximum
travel distance and parcel size. In contrast, a truck has the advantage of long range travel capability. It can carry large and heavy
cargo with a diversity of size, but it is also heavy, slow and has much higher transportation cost.
Consequently, the advantages of truck offset the disadvantages of drones and—similarly–the advantages of drones offset the disadvantages
of trucks. These complementary capabilities are the foundation of a novel method named “last mile delivery with drone”
(Banker, 2013), in which the truck transports the drone close to the customer locations, allowing the drone to service customers while
remaining within its flight range, effectively increasing the usability and making the schedule more flexible for both drone The MILP
formulation is as follows and trucks. Specifically, a truck departs from the depot carrying the drone and all the customer parcels. As the
truck makes deliveries, the drone is launched from the truck to service a nearby customer with a parcel. While the drone is in service, the
truck continues its route to further customer locations. The drone then returns to the truck at a location different from its launch point.
From the application perspective, a number of remarkable events have occurred since 2013, when Amazon CEO Jeff Bezos first
announced Amazon’s plans for drone delivery (News, 2013), termed “a big surprise.” Recently, Google has been awarded a patent that
outlines its drone delivery method (Murphy, 2016). In detail, rather than trying to land, the drone will fly above the target, slowly
lowering packages on a tether. More interestingly, it will be able to communicate with humans using voice messages during the delivery
process. Google initiated this important drone delivery project, called Wing, in 2014, and it is expected to launch in 2017 (Grothaus,
2016). A similar Amazon project called Amazon Prime Air ambitiously plans to deliver packages by drone within 30 min (Pogue, 2016).
Other companies worldwide have also been testing delivery services using drones. In April 2016, Australia Post successfully tested drones
for delivering small packages. That project is reportedly headed towards a full customer trial in late 2016 (Cuthbertson, 2016). In May
2016, a Japanese company—Rakuten—launched a service named “Sora Kaku” that “delivers golf equipment, snacks, beverages and other
items to players at pickup points on the golf course” (News, 2016). In medical applications,Matternet, a California-based startup, has been
testing drone deliveries of medical supplies and specimens (such as blood samples) in many countries since 2011. According to their CEO:
it is “much more cost-, energy- and time-efficient to send [a blood sample] via drone, rather than send it in a two-ton car down the
highway with a person inside to bring it to a different lab for testing,” (French, 2015). Additionally, a Silicon Valley start-up named Zipline
International began using drones to deliver medicine in Rwanda starting in July, 2016 (Toor, 2016).
We are aware of several publications in the literature that have investigated the routing problem related to the truck-drone
combination for delivery. Murray and Chu (2015) introduced the problem, calling it the “Flying Sidekick Traveling Salesman Problem”
(FSTSP). A mixed integer liner programming (MILP) formulation and a heuristic are proposed. Basically, their heuristic is
based on a “Truck First, Drone Second” idea, in which they first construct a route for the truck by solving a TSP problem and, then,
repeatedly run a relocation procedure to reduce the objective value. In detail, the relocation procedure iteratively checks each node
from the TSP tour and tries to consider whether it is suitable for use as a drone node. The change is applied immediately when this is
true, and the current node is never checked again. Otherwise, the node is relocated to other positions in an effort to improving the
objective value. The relocation procedure for TSP-D is designed in a “best improvement” fashion; it evaluates all the possible moves
and executes the best one. The proposed methods are tested only on small-sized instances with up to 10 customers.
Agatz et al. (2016), study a slightly different problem—called the “Traveling Salesman Problem with Drone” (TSP-D), in which
the drone has to follow the same road network as the truck. Moreover, in TSP-D, the drone may be launched and return to the same
location, while this is forbidden in the FSTSP. This problem is also modelled as a MILP formulation and solved by a “Truck First,
Drone Second” heuristic in which drone route construction is based on either local search or dynamic programming. Recently,
Bouman et al. (2017) extended this work by proposing an exact approach based on dynamic programming that is able to solve larger
instances. Furthermore, Ponza (2016) also extended the work of Murray and Chu (2015) in his master’s thesis to solve the FSTSP,
proposing an enhancement to the MILP model and solving the problem by a heuristic method based on Simulated Annealing.
Additionally, Wang et al. (2016), in a recent research, introduced a more general problem that copes with multiple trucks and
drones with the goal of minimizing the completion time. The authors named the problem “The vehicle routing problem with drone”
(VRP-D) and conducted the analysis on several worst-case scenarios, from which they propose bounds on the best possible savings in
time when using drones and trucks instead of trucks alone. A further development of this research is studied in Poikonen et al. (2017)
where the authors extend the worst-case bounds to more generic distance/cost metrics as well as explicitly consider the limitation of
battery life and cost objectives.
All the works mentioned above aim to minimize the time at which the truck and the drone complete the route and return to the depot,
which can improve the quality of service (Nozick and Turnquist, 2001). However, in every logistics activity, operational costs also play an
important role in the overall business cost (see Russell et al., 2014; Robinson, 2014). Hence, minimizing these costs by using a more costefficient
approach is a vital objective of every company involved in transport and logistics activities. Recently, an objective function that
minimizes the transportation cost was studied by Mathew et al. (2015) for a related problem called the Heterogeneous Delivery Problem
(HDP). However, unlike in Murray and Chu (2015) and Agatz et al. (2016), the problem is modelled on a directed physical street network
where a truck cannot achieve direct delivery to the customer. Instead, from the endpoint of an arc, the truck can launch a drone that will
service the customers. In this way, the problem can be favourably transformed to a Generalized Traveling Salesman Problem (GTSP)
(Gutin and Punnen, 2006). The authors use the Nood-Bean Transformation available in Matlab to reduce a GTSP to a TSP, which is then
solved by a heuristic proposed in the study. To the best of our knowledge, the min-cost objective function has not been studied for TSP-D
when the problem is defined in a more realistic way—similarly to Murray and Chu (2015) and Agatz et al. (2016). Consequently, this gap
in the literature provides a motivation for studying TSP-D with the min-cost objective function.
This paper studies a new variant of TSP-D following the hypotheses of the FSTSP proposed in the work of Murray and Chu (2015). In
FSTSP, the objective is to minimize the delivery completion time, or in other word the time coming back to the depot, of both truck and
drone. In the new variant that we call min-cost TSP-D, the objective is to minimize the total operational cost of the system including two
distinguished parts. The first part is the transportation cost of truck and drone while the second part relates to the waste time a vehicle has
to wait for the other whenever drone is launched. In the following, we denote the FSTSP as min-time TSP-D to avoid confusion.
In this paper, we propose a MILP model and two heuristics to solve the min-cost TSP-D: a Greedy Randomized Adaptive Search
Procedure (GRASP) and a heuristic adapted from the work of Murray and Chu (2015) called TSP-LS. In detail, the contributions of
this paper are as follows:
– We introduce a new variant of TSP-D called min-cost TSP-D, in which the objective is to minimize the operational costs.
– We propose a model together with a MILP formulation for the problem which is an extended version of the model proposed in
Murray and Chu (2015) for min-time TSP-D.
– We develop two heuristics for min-cost TSP-D: TSP-LS and GRASP. which contain a new split procedure and local search operators.
We also adapt our solution methods to solve the min-time problem studied in Murray and Chu (2015).
– We introduce various sets of instances with different numbers of customers and a variety of options to test the problem.
– We conduct various experiments to test our heuristics on the min-cost as well as min-time problems. We also compare solutions of
both objectives. The computational results show that GRASP outperforms TSP-LS in terms of quality of solutions with an acceptable
running time. TSP-LS delivers solution of lower quality, but very quickly.
This article is structured as follows: Section 1 provides the introduction. Section 2 describes the problem and the model. The MILP
formulation is introduced in Section 3. We describe our two heuristics in Sections 4 and 5. Section 6 presents the experiments,
including instance generations and settings. We discuss the computational results in Section 7. Finally, Section 8 concludes the work
and provides suggestions for future research.

## 2. Problem definition

In this section, we provide a description of the problem and discuss a model for the min-cost TSP-D in a step-by-step manner.
Here, we consider a list of customers to whom a truck and a drone will deliver parcels. To make a delivery, the drone is launched from
the truck and later rejoins the truck at another location. Each customer is visited only once and is serviced by either the truck or the
drone. Both vehicles must start from and return to the depot. When a customer is serviced by the truck, this is called a truck delivery,
while when a customer is serviced by the drone, this is called a drone delivery. This is represented as a 3-tuple 〈i,j,k〉, where i is a
launch node, j is a drone node (a customer who will be serviced by the drone), and k is a rendezvous node, as listed below:
• Node i is a launch node at which the truck launches the drone. The launching operation must be carried out at a customer location
or the depot. The time required to launch the drone is denoted as sL.
• Node j is a node serviced by the drone, called a “drone node”. We also note that not every node in the graph is a drone node.
Because some customers might demand delivery a product with size and weight larger than the capacity of the drone.
• Node k is a customer location where the drone rejoins the truck. At node k, the two vehicles meet again; therefore, we call it
“rendezvous node”. While waiting for the drone to return from a delivery, the truck can make other truck deliveries. The time
required to retrieve the drone and prepare for the next drone delivery is denoted as sR. Moreover, the two vehicles can wait for
each other at the rendezvous point.
Moreover, the drone has an “endurance”, which can be measured as the maximum time the drone can operate without recharging.
A tuple 〈i,j,k〉 is called feasible if the drone has sufficient power to launch from i, deliver to j and rejoin the truck at k. The drone can
be launched from the depot but must subsequently rejoin the truck at a customer location. Finally, the drone’s last rendezvous with
the truck can occur at the depot.
When not actively involved in a drone delivery, the drone is carried by the truck. Again, as in Murray and Chu (2015), we also
assume that the drone is in constant flight when waiting for the truck. Furthermore, the truck and the drone have their own
transportation costs per unit of distance. In practice, the drone’s cost is much lower than the truck’s cost because it weighs much less
than the truck, hence, consuming much less energy. In addition, it is not run by gasoline but by batteries. We also assume that the
truck provides new fresh batteries for the drone (or recharges its batteries completely) before each drone delivery begins. When a
vehicle has to wait for each other, a penalty is created and added to the transportation cost to form the total operational cost of the
system. The waiting costs of truck and drone are calculated by:
waiting costtruck = α × waiting time andwaiting costdrone = β × waiting time
where α and β are the waiting fees of truck and drone per unit of time, respectively. The use of these coefficients is flexible to model a
number of situations in reality. Typically, α can be used to represent parking fee and labour cost of the truck driver, while β can
model costs resulted by battery energy consumption of drone. In some contexts where a delivery company does not have to pay
waiting costs and wants to focus only on transportation cost, it can set both coefficients to null. In addition, if we do not allow the
waiting of drone to protect it from theft or being shot down by strangers, we can set β to a very large number. Similarly for α, if we
only have a very short time to park the truck at the customer locations.
The objective of the min-cost TSP-D is to minimize the total operational cost of the system which includes the travel cost of truck
and drone as well as their waiting costs. Because the problem reduces to a TSP when the drone’s endurance is null, it is NP-Hard.
Examples of TSP and min-cost TSP-D optimal solutions on the same instance in which the unitary transportation cost of the truck is 25
times more expensive than that of the drone and α, β both are set to null are shown in Fig. 1.
We now develop the model for the problem. We first define basic notations relating to the graph, sequence and subsequence.
Then, we formally define drone delivery and the solution representation as well as the associated constraints and objective.


![Fig. 1. Optimal solution: TSP vs. min-cost TSP-D. TSP Objective=1500, min-cost TSP-D Objective=1000.82. The solid arcs are truck’s path. The dash arcs are
drone’s path.](https://github.com/vanherzog/TSPD/blob/master/Figure%201.png)
