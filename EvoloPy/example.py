# -*- coding: utf-8 -*-

from optimizer import run
from dataset import read_graph
# Select optimizers
# "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS","HHO","SCA","JAYA","DE"
optimizer=["FFA"]

# Select benchmark function"
# "F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19"
objectivefunc=["F_1"] 

# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
NumOfRuns=1

# Select general parameters for all optimizers (population size, number of iterations) ....
params = {'PopulationSize' : 5, 'Iterations' : 50}

#Choose whether to Export the results in different formats
export_flags = {'Export_avg':False, 'Export_details':False, 
'Export_convergence':False, 'Export_boxplot':False}

a,b = read_graph()
data_package = {"centrality":b, "shortest":a}
run(optimizer, objectivefunc, NumOfRuns, params, export_flags, data_package)
