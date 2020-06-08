# -*- coding: utf-8 -*-

from __future__ import division
import time
import random
import numpy
import math
from solution import solution
import networkx as nx

N=0 #global variable to save number of nodes in a graph
cent = {}
GlobalMax, GlobalLeaderPosition, GlobalLimitCount, LocalMax, LocalLimitCount, LocalLeaderPosition = None, None, None, None, None, None 
G = None 

def read_graph():
    global G
    G = nx.read_gml("/home/raj/Desktop/research_papers/SpiderMonkey/EvoloPy/netscience.gml")
    G = nx.convert_node_labels_to_integers(G,first_label=1)
    max_, nodes  = 0, {}
    for item in nx.connected_components(G):
        if len(item)>max_:
            max_ = len(item)
            nodes = item 
    
    G = nx.subgraph(G, set(nodes))
    G = nx.convert_node_labels_to_integers(G,first_label=1)
   
def find_sort_neighbors(inpt_array):
    """ return a dictionary of neigbours of each node in inpt_array """
    main_lst = {}
    for n in inpt_array:
        neig = list(G.neighbors(n))
        main_lst[n] = neig
    
    return main_lst


class SMO():
    def __init__(self,objf1,lb1,ub1,dim1,PopSize1,acc_err1,iters1):
        self.PopSize=PopSize1
        self.dim=dim1
        self.acc_err=acc_err1
        self.lb=lb1
        self.ub=ub1
        self.objf=objf1
        self.pos=numpy.zeros((PopSize1,dim1), dtype=int)
        self.fun_val = numpy.zeros(PopSize1)
        self.fitness = numpy.zeros(PopSize1)
        # Assuming all population are 1 group. So lo=0 and hi= population -1, for gpoint[0:]
        self.group = 1
        self.gpoint = numpy.zeros((PopSize1,2), dtype=int)
        self.gpoint[0,0] = 0
        self.gpoint[0,1] = PopSize1-1 
        self.prob=numpy.zeros(PopSize1)
        self.LocalLimit=dim1*PopSize1
        self.GlobalLimit=PopSize1;
        self.fit = numpy.zeros(PopSize1)
        self.MinCost=numpy.zeros(iters1)
        self.Bestpos=numpy.zeros(dim1, dtype = int)
        self.func_eval=0
        self.part=1
        self.max_part=PopSize1/10
        self.cr=0.1

    def change_coordinate(self, original_co, leader_co, rand_co, dim, cr=0.5, subtr=0.5, mult=2):
        
        inter_local = numpy.in1d(original_co, leader_co)
        inter_local = numpy.where(inter_local==True, 0, 1)
        inter_random = numpy.in1d(original_co, rand_co)
        inter_random = numpy.where(inter_random==True,0,1)

        r1 = random.random()
        r2 = (random.random()-subtr)*mult
        l_final = inter_local*r1 + inter_random*r2 
        decision = numpy.where(l_final<cr, 0, 1)
        
        final_co=numpy.zeros(dim)
        for i in range(dim):
            if decision[i]==0:
                final_co[i] = original_co[i]
            else: # replace it 
                random_coordinate = 1
                while True:
                    if  type(self.ub)==int:
                        random_coordinate = random.randint(self.lb,self.ub)
                    else:
                        random_coordinate = random.randint(self.lb[i], self.ub[i])

                    if random_coordinate not in final_co:
                        break 
                final_co[i] = random_coordinate
        
        return final_co
        

    # ====== Function: CalculateFitness() ========= #
    def CalculateFitness(self,fun1):
        if fun1 >= 0:
        #     result = (1/(fun1+1))
            result = fun1 
        else:
            # result=(1+math.fabs(fun1))
            result = math.fabs(fun1)
            
        return result
    #================ X X X ===================== #

    # ==================================== Function: Initialization() ============================================ #
    def initialize(self):
        global GlobalMax, GlobalLeaderPosition, GlobalLimitCount, LocalMax, LocalLimitCount, LocalLeaderPosition
        # S_max=int(self.PopSize/2)
        S_max = self.max_part # max number of groups allowed
        LocalMax = numpy.zeros(S_max)
        LocalLeaderPosition=numpy.zeros((S_max,self.dim), dtype = int)
        LocalLimitCount=numpy.zeros(S_max, dtype = int)
        for i in range(self.PopSize):
            for j in range(self.dim):
                if type(self.ub)==int:
                    self.pos[i,j]=random.randint(self.lb, self.ub) # select a node between node 1 and nth node ( here 379)
                else:
                    self.pos[i,j]=random.randint(self.lb[j],self.ub[j])

        #Calculate objective function for each particle
        for i in range(self.PopSize):
            # Performing the bound checking
            # self.pos[i,:]=numpy.clip(self.pos[i,:], self.lb, self.ub)
            self.fun_val[i]=self.objf(self.pos[i,:], cent)
            self.func_eval+=1
            self.fitness[i]=self.CalculateFitness(self.fun_val[i])

        # Initialize Global Leader Learning
        GlobalMax=self.fun_val[0]
        GlobalLeaderPosition=self.pos[0,:]
        GlobalLimitCount=0

        # Initialize Local Leader Learning
        for k in range(self.group): # k will always be = 0 during initialization
            LocalMax[k]=self.fun_val[int(self.gpoint[k,0])]
            LocalLimitCount[k]=0
            LocalLeaderPosition[k,:]=self.pos[int(self.gpoint[k,0]),:]
    # ============================================ X X X ======================================================= #


    # =========== Function: CalculateProbabilities() ============ #
    def CalculateProbabilities(self):
        maxfit=self.fitness[0];
        i=1
        while(i<self.PopSize):
            if (self.fitness[i]>maxfit):
                maxfit=self.fitness[i];
            i+=1
        for i in range(self.PopSize):
            self.prob[i]=(0.9*(self.fitness[i]/maxfit))+0.1;
    # ========================== X X X ======================== #

    # ================= Function: create_group() ================ #
    def create_group(self):
        g=0
        lo=0
        while(lo < self.PopSize):
            hi= lo+int(self.PopSize/self.part)
            self.gpoint[g,0]=lo
            self.gpoint[g,1]=hi
            if((self.PopSize-hi)<(int(self.PopSize/self.part))):
                self.gpoint[g,1]=(self.PopSize-1)
            g=g+1
            lo=hi+1
        self.group = g
    # ========================== X X X ======================== #

    # ================= Function: LocalLearning() ================ #
    def LocalLearning(self):
        global LocalMax, LocalLimitCount, LocalLeaderPosition
        S_max=self.max_part # max number of groups possible is when each group has 2 member. 
        OldMin = numpy.zeros(S_max)
        for k in range(self.group):
            OldMin[k]=LocalMax[k]

        for  k in range(self.group):
            i=int(self.gpoint[k,0])
            while (i<=int(self.gpoint[k,1])):
                # Assuming local leader will have highest function value which is also same as fitness value
                if (self.fun_val[i]>LocalMax[k]): 
                    LocalMax[k]=self.fun_val[i]
                    LocalLeaderPosition[k,:]=self.pos[i,:]
                i=i+1
       
        for k in range(self.group):
            # if (math.fabs(OldMin[k]-LocalMax[k])<self.acc_err):
            if LocalMax[k] < (self.acc_err + OldMin[k]): # LocalMax should increase by more than accepted error(here, 0.00001) from Old local max value    
                LocalLimitCount[k]=LocalLimitCount[k]+1
            else:
                LocalLimitCount[k]=0
    # ========================== X X X ======================== #

    # ================= Function: GlobalLearning() ================ #
    def GlobalLearning(self):
        global GlobalMax, GlobalLeaderPosition, GlobalLimitCount
        G_trial=GlobalMax
        for i in range(self.PopSize):
            if (self.fun_val[i] > GlobalMax):
                GlobalMax=self.fun_val[i]
                GlobalLeaderPosition=self.pos[i,:]

        # if(math.fabs(G_trial-GlobalMax)<self.acc_err):
        if GlobalMax < (self.acc_err + G_trial):
            GlobalLimitCount=GlobalLimitCount+1
        else:
            GlobalLimitCount=0
    # ========================== X X X ======================== #

    # ================= Function: LocalLeaderPhase() ================ #
    def LocalLeaderPhase(self,k):
        global LocalLeaderPosition
        new_position=numpy.zeros((1,self.dim))
        lo=int(self.gpoint[k,0])
        hi=int(self.gpoint[k,1])
        i=lo
        while(i <=hi):
            while True:
                # PopRand=int((random.random()*(hi-lo)+lo))
                PopRand = random.randint(lo, hi) # select a member within the group other that myself
                if (PopRand != i):
                    break
            # for j in range(self.dim):
            #     if (random.random() >= self.cr):
            #         new_position[0,j]=self.pos[i,j]+(LocalLeaderPosition[k,j]-self.pos[i,j])*(random.random())+(self.pos[PopRand,j]-self.pos[i,j])*(random.random()-0.5)*2
            #     else:
            #         new_position[0,j]=self.pos[i,j]
            # new_position=numpy.clip(new_position, self.lb, self.ub)

            intersection_local = numpy.isin(self.pos[i,:], LocalLeaderPosition[k,:]) # all those elements in pos[i,:] which are also in LocalLeaderPosition will be true, all else will be false
            intersection_local = numpy.where(intersection_local==True, 0, 1) # make all existing values as 0 which means they don't have to be replaced

            intersection_random = numpy.isin(self.pos[i,:], self.pos[PopRand,:]) 
            if -1+random.random()*(1-(-1))>=0:
                #I want to be like that random monkey of my group
                intersection_random = numpy.where(intersection_random==True, 0, 1) # replace all values which don't match with him
            else:
                #I strictly don't want to be like that random monkey of my group in any way!!!.
                intersection_random = numpy.where(intersection_random==True, 1, 0) # replace all values which match with him


            r1,r2= [random.random() for _ in range(2)]
            final_array = intersection_local*r1 + intersection_random*r2 
            final_array = numpy.where(final_array>=self.cr, True, False) #True values will replace when final random sum >= pr

            # new_position = self.pos[i,:]
            # new_position = new_position[final_array]
            neighbors = find_sort_neighbors(self.pos[i,:][final_array])
            new_position = self.pos[i,:]
            while True:
                temp = new_position
                for j in range(len(final_array)):
                    if  final_array[j]==True:
                        temp[j] = random.choice(neighbors[self.pos[i,j]])
                if len(numpy.unique(temp)) == len(new_position):
                    new_position = temp
                    break


            
            ObjValSol=self.objf(new_position, cent)
            self.func_eval+=1
            FitnessSol=self.CalculateFitness(ObjValSol)
            if (FitnessSol>self.fitness[i]):
                self.pos[i,:]=new_position
                self.fun_val[i]=ObjValSol
                self.fitness[i]=FitnessSol
            i+=1
    # ========================== X X X ======================== #

    # ================= Function: GlobalLeaderPhase() ================ #
    def GlobalLeaderPhase(self,k):
        global GlobalLeaderPosition
        new_position=numpy.zeros((1,self.dim))
        lo=int(self.gpoint[k,0])
        hi=int(self.gpoint[k,1])
        i=lo;
        leh=lo;
        while(leh<hi):
            # if (random.random() < self.prob[i]):
            #     leh+=1
            while True:
                PopRand=random.randint(lo,hi)
                if (PopRand != i):
                    break
                
            # param2change=int(random.random()*self.dim)
            
            # new_position[param2change]=self.pos[i,param2change]+(GlobalLeaderPosition[param2change]-self.pos[i,param2change])*(random.random())+(self.pos[PopRand,param2change]-self.pos[i,param2change])*(random.random()-0.5)*2
            intersection_global = numpy.isin(self.pos[i,:], GlobalLeaderPosition) # all those elements in pos[i,:] which are also in GlobalLeaderPosition will be true, all else will be false
            intersection_global = numpy.where(intersection_global==True, 0, 1) # make all existing values as 0 which means they don't have to be replaced
            intersection_random = numpy.isin(self.pos[i,:], self.pos[PopRand,:]) 
            if -1+random.random()*(1-(-1))>=0:
                intersection_random = numpy.where(intersection_random==True, 0, 1) # replace all values not matching with him
            else:
                intersection_random = numpy.where(intersection_random==True, 1, 0) # replace all values matching with him

            r1,r2= [random.random() for _ in range(2)]
            final_array = intersection_global*r1 + intersection_random*r2 
            final_array = numpy.where(final_array>=self.prob[i], True, False)

            change_array = self.pos[i,:][final_array]
            neighbors = find_sort_neighbors(change_array)
            new_position=self.pos[i,:]
            
            while True:
                temp = new_position
                for j in range(len(final_array)):
                    if  final_array[j]==True:
                        temp[j] = random.choice(neighbors[self.pos[i,j]])
                if len(numpy.unique(temp)) == len(new_position):
                    new_position = temp
                    leh+= len(change_array)
                    break

            # new_position=numpy.clip(new_position, self.lb, self.ub)
            ObjValSol=self.objf(new_position, cent)
            self.func_eval+=1
            FitnessSol=self.CalculateFitness(ObjValSol)
            if (FitnessSol>self.fitness[i]):
                self.pos[i,:]=new_position
                self.fun_val[i]=ObjValSol
                self.fitness[i]=FitnessSol
            i+=1;
            if i==hi:
                i=lo;
    # ========================== X X X ======================== #

    # ================= Function: GlobalLeaderDecision() ================ #
    def GlobalLeaderDecision(self):
        global GlobalLimitCount
        if(GlobalLimitCount> self.GlobalLimit):
            GlobalLimitCount=0
            
            if(self.part<self.max_part):
                self.part=self.part+1
                # self.create_group()
                # self.LocalLearning()
            else:
                self.part=1
            
            self.create_group()
            self.LocalLearning()
    # ========================== X X X ======================== #

    # ================= Function: LocalLeaderDecision() ================ #
    def LocalLeaderDecision(self):
        global GlobalLeaderPosition, LocalLimitCount, LocalLeaderPosition
        for k in range(self.group):
            if(LocalLimitCount[k]>self.LocalLimit):
                i=self.gpoint[k,0]
                while(i<=int(self.gpoint[k,1])):
                    # for j in range(self.dim):
                        # if (random.random()>= self.cr):
                        #     if type(self.ub)==int:
                        #         self.pos[i,j]=random.random()*(self.ub-self.lb)+self.lb
                        #     else:
                        #         self.pos[i,j]=random.random()*(self.ub[j]-self.lb[j])+self.lb[j]
                        # else:
                        #     self.pos[i,j]=self.pos[i,j]+(GlobalLeaderPosition[j]-self.pos[i,j])*random.random()+(self.pos[i,j]-LocalLeaderPosition[k,j])*random.random()
                    
                    intersection_global = numpy.isin(self.pos[i,:], GlobalLeaderPosition) # all those elements in pos[i,:] which are also in GlobalLeaderPosition will be true, all else will be false
                    intersection_global = numpy.where(intersection_global==True, 0, 1) # make all existing values as 0 which means they don't have to be replaced
                    intersection_local = numpy.isin(self.pos[i,:], LocalLeaderPosition[k,:]) # all those elements in pos[i,:] which are also in LocalLeaderPosition will be true, all else will be false
                    intersection_local = numpy.where(intersection_local==True, 1, 0) # All matching nodes would be set to 1 i.e. replaced.

                    r1,r2= [random.random() for _ in range(2)]
                    final_array = intersection_global*r1 + intersection_local*r2
                    new_position = self.pos[i,:]
                    change_array = []     
                    for j in range(len(self.pos[i,:])):
                        if final_array[j] >= self.cr:
                            while True:
                                new_node = random.randint(self.lb, self.ub)
                                if new_node not in new_position:
                                    new_position[j] = new_node
                                    break 
                        else:
                            change_array.append(new_position[j])
                    
                    neighbors = find_sort_neighbors(change_array)
                    while True:
                        temp = new_position
                        for j in range(len(final_array)):
                            if  final_array[j]==change_array[j]:
                                temp[j] = random.choice(neighbors[self.pos[i,j]])
                        if len(numpy.unique(temp)) == len(new_position):
                            new_position = temp
                            break

                    # self.pos[i,:]=numpy.clip(self.pos[i,:], self.lb, self.ub)
                    self.fun_val[i]=self.objf(self.pos[i,:])
                    self.func_eval+=1
                    self.fitness[i]=self.CalculateFitness(self.fun_val[i])
                    i+=1
                LocalLimitCount[k]=0
    # ========================== X X X ======================== #



# ==================================== Main() ===================================== #
def main(objf1,lb1,ub1,dim1,PopSize1,iters,acc_err1,obj_val,succ_rate,mean_feval, data_dict, Num_nodes):
    global N, cent 
    cent = data_dict["centrality"]
    N = Num_nodes
    read_graph()
    smo=SMO(objf1,lb1,ub1,dim1,PopSize1,acc_err1,iters)
    s=solution()
    print("SMO is optimizing  \""+smo.objf.__name__+"\"")    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")

    # =========================== Calling: initialize() =========================== #
    smo.initialize(data_dict)

    # ========================== Calling: GlobalLearning() ======================== #
    smo.GlobalLearning()

    # ========================= Calling: LocalLearning() ========================== #
    smo.LocalLearning()

    # ========================== Calling: create_group() ========================== #
    smo.create_group()

    # ================================= Looping ================================== #
    for l in range(iters):
        for k in range(smo.group-1):
            # ==================== Calling: LocalLeaderPhase() =================== #
            smo.LocalLeaderPhase(k)
            
        # =================== Calling: CalculateProbabilities() ================== #
        smo.CalculateProbabilities()

        for k in range(smo.group-1):
            # ==================== Calling: GlobalLeaderPhase() ================== #
            smo.GlobalLeaderPhase(k)
            
        # ======================= Calling: GlobalLearning() ====================== #
        smo.GlobalLearning()

        # ======================= Calling: LocalLearning() ======================= #
        smo.LocalLearning()

        # ================== Calling: LocalLeaderDecision() ====================== #
        smo.LocalLeaderDecision()

        # ===================== Calling: GlobalLeaderDecision() ================== #
        smo.GlobalLeaderDecision()

        # ======================= Updating: 'cr' parameter ======================= #
        smo.cr = smo.cr + (0.4/iters)
        
        # ====================== Saving the best individual ====================== #        
        smo.MinCost[l] = GlobalMax
        Bestpos=smo.pos[1,:]
        gBestScore=GlobalMax


        # ================ Displaying the fitness of each iteration ============== #        
        if (l%1==0):
               print(['At iteration '+ str(l+1)+ ' the best fitness is '+ str(gBestScore)])
               print("Best position: ", Bestpos)
               print()

        # ====================== Checking: acc_error ============================ #        
        if(math.fabs(GlobalMax-obj_val)<=smo.acc_err):
            succ_rate+=1
            mean_feval=mean_feval+smo.func_eval
            break
    # ========================= XXX Ending of Loop XXX ========================== #        

    # =========================== XX Result saving XX =========================== #
    error1=math.fabs(GlobalMax-obj_val)
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=smo.MinCost
    s.optimizer="SMO"
    s.error = error1
    s.feval=smo.func_eval
    s.objfname=smo.objf.__name__

    return s, succ_rate,mean_feval

    # ================================ X X X =================================== #
         
    
