""" To read graph and return their centrality figures """

import networkx as nx 
from statistics import mean

def read_graph():
    """ Returns dictionary of shortest paths and additive centrality(deg/mean(CC,EC,PC,BC)) """

    G = nx.read_gml("/home/raj/Desktop/research_papers/SpiderMonkey/netscience.gml")
    print(nx.info(G))

    cent, node_map = find_centralities(G)
    shortest = dict(nx.all_pairs_dijkstra_path_length(G,weight="value"))
    new_shortest = {}
    for src, dic in shortest.items():
        temp_dic = {}
        for node, v in dic.items():
            temp_dic[node_map[node]]=v 

        new_shortest[node_map[src]] = temp_dic

    return new_shortest, cent

def find_centralities(G):

    deg = nx.degree_centrality(G)
    closen = nx.closeness_centrality(G,distance="value")
    eigen = nx.eigenvector_centrality(G, weight="value")
    betwn = nx.betweenness_centrality(G,weight="value")
    page = nx.pagerank(G,weight="value")

    additive_cent={}
    node_map={}
    i=1
    for n in G.nodes():
        theta = (1+mean([closen[n], eigen[n], betwn[n], page[n]]))/(1+deg[n])
        additive_cent[i]=theta 
        node_map[n]=i 
        i+= 1

    return additive_cent, node_map

if __name__ == "__main__":
    short, cent = read_graph()
    print(short)