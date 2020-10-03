## Descripion :
NOTE: The programs has hardcoded directory path. Do change them according to your own local paths. <3 
<hr> 

### 1. Netscience.gml (gml : Graph Markup Language)
 The file containing adjacency list of our sample network. This network refers to the authorship graphs of who has 
 worked with whom in the scientific community. 

###  2. dataset.py 
 This program reads the gml file and produces a file named "Netscience.json" which contains influence of a node mapped to the node ID. The influence is calculated using a combination of a few of the conventional functions called as "Centrality Functions".

### 3. SMO.py 
The Spider Monkey program fitted to work on a given graph. 
 
### 4. benchmark.py 
This file contains the Function which we want to optimize using our Swarm algo. F_1 is the function of our focus. Given a list of nodes -> x and a dictionary where nodes are mapped to their influence score, the function F_1 returns the overall score of the input list of nodes. Function "getfunctiondetails" has the lower bound, upper bound, accepted error and some other parameters useful to the algo. Bound values helps to choose the next random node to visit, while accepted error defines how much close to the objective value the optimizer needs to get before terminating. 

### 5. solution.py
Defines a class having all necessary data of all solutions generated in the algo. 

### 6. Main.py 
Runs the SMO.py and displays the final results. 