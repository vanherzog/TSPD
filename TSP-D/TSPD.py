import random

class TSPD:
    def grasp(self):
        bestSolution = None
        places = 0 #List of all places
        bestObjectiveValue = [] #List of possible Drone deliveries and costs
        randomGenerator = self.randomGenerator(places)
        iteration = 0

        while(iteration < 100): #nTSP für 100
            iteration = iteration + 1
            tour = self.randomGenerator(places)


    def randomGenerator(self, places):
        return random.shuffle(places)
        #hat geklappt