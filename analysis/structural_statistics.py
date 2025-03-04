#compute statistics on a set of graphs
import itertools

import numpy as np
import networkx as nx


def check_out(graphs):
    #compute statistics on a set of graphs
    #graphs: list of networkx graphs
    #return: pandas dataframe with statistics
    stats = []
    flag = False
    for i, f in enumerate(graphs):
        #convert each multidirected graph to a wighted directed graph, with weights equal to the number of edges and keep the labels of the nodes
        g = nx.DiGraph()
        g.add_nodes_from(f.nodes(data=True))
        g.add_edges_from(f.edges(data=True))
        for node in g.nodes():
            if g.out_degree(node) == 0:
                flag = True
                break
        if flag:
            return flag

    return flag

def compute_global_statistics(graphs):
    #compute statistics on a set of graphs
    #graphs: list of networkx graphs
    #return: pandas dataframe with statistics
    stats = []
    for i, f in enumerate(graphs):
        #convert each multidirected graph to a wighted directed graph, with weights equal to the number of edges and keep the labels of the nodes

        g = nx.DiGraph()
        g.add_nodes_from(f.nodes(data=True))
        g.add_edges_from(f.edges(data=True))


        g_undirect = g.to_undirected()
        #create a fully connected graph with the nodes from the original graph
        g2 = nx.DiGraph()

        couples = list(itertools.combinations(g.nodes(), 2))
        couples_rev = [(v, u) for u, v in couples]

        g2.add_edges_from([(u, v, {"weight": 0}) for u, v in couples+couples_rev])

        for u, v, d in g2.edges(data=True):
            d["weight"] = f.number_of_edges(u, v)

        for u, v, d in g.edges(data=True):
            d["weight"] = f.number_of_edges(u, v)

        count_cliques2 = 0
        tot = 0
        for couple in itertools.combinations(g.nodes(), 2):
            u, v = couple
            if g.has_edge(v, u) and g.has_edge(v, u):
                count_cliques2 += 1
            tot += 1

        count_cliques2 = count_cliques2/tot

        count_back_and_forth = 0
        tot = 0
        for couple in itertools.combinations(g2.nodes(), 2):
            u, v = couple
            if g2[u][v]["weight"] > 1 and g2[v][u]["weight"] >1:
                count_back_and_forth += 1
            tot += 1
        count_back_and_forth = count_back_and_forth/tot

        avg_degree = np.mean([g_undirect.degree(node) for node in g_undirect.nodes()])

        avg_out_degree = np.mean([g.out_degree(node) for node in g.nodes()])

        avg_in_degree = np.mean([g.in_degree(node) for node in g.nodes()])

        #create a dictionary with the statistics
        stats.append({"graph": i, "Cycles": count_cliques2, "Consistent Cycles": count_back_and_forth, "Average Degree": avg_degree/g.number_of_nodes(), "Average In Degree":avg_in_degree/g.number_of_nodes(), "Average Out Degree":avg_out_degree/g.number_of_nodes(), "Transitivity": nx.transitivity(g_undirect)})


    return stats
