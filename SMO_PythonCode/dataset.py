""" To read graph and return their centrality figures """

import networkx as nx 
from statistics import mean
import json 

def read_graph():
    """ Returns dictionary of shortest paths and additive centrality(deg/mean(CC,EC,PC,BC)) """

    G = nx.read_gml("/home/raj/Desktop/research_papers/SpiderMonkey/EvoloPy/netscience.gml")
    G = nx.convert_node_labels_to_integers(G,first_label=1)
    max_, nodes  = 0, {}
    for item in nx.connected_components(G):
        if len(item)>max_:
            max_ = len(item)
            nodes = item 
    
    G = nx.subgraph(G, set(nodes))
    G = nx.convert_node_labels_to_integers(G,first_label=1)
    print(nx.info(G))

    cent = find_centralities(G)
    shortest = dict(nx.all_pairs_dijkstra_path_length(G,weight="value"))
    # new_shortest = {}
    # for src, dic in shortest.items():
        # temp_dic = {}
        # for node, v in dic.items():
            # temp_dic[node_map[node]]=v 

        # new_shortest[node_map[src]] = temp_dic
    main_dict = {}
    main_dict["shortest"]=shortest
    main_dict["centrality"] = cent 
    json.dump(main_dict, open("Netscience.json", "w"))
    return shortest, cent

def find_centralities(G):

    deg = nx.degree_centrality(G)
    closen = nx.closeness_centrality(G,distance="value")
    eigen = nx.eigenvector_centrality(G, weight="value")
    betwn = nx.betweenness_centrality(G,weight="value")
    page = nx.pagerank(G,weight="value")

    additive_cent={}
    # node_map={}
    # i=1
    for n in G.nodes():
        theta = (mean([closen[n], eigen[n], betwn[n], page[n]]))/(1+deg[n])
        additive_cent[n]=theta 
        # node_map[n]=i 
        # i+= 1

    return additive_cent

if __name__ == "__main__":
    short, cent = read_graph()
    # print(cent)