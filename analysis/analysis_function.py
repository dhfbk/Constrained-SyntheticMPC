import json

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
import statsmodels.api as sm

import warnings

def check_correct_format(datalist):

    good = 0

    good_list = []

    for data in datalist:

        flag = True

        try:
            speakers = []
            test = json.loads(data)
            #check if the json has the correct format
            if "conversation" not in test.keys() or "speakers" not in test.keys():
                flag = False
            else:
                for speaker in test["speakers"]:
                    if "name" not in speaker.keys() or "stance" not in speaker.keys():
                        flag = False
                    else:
                        if speaker["stance"].lower() not in ["positive", "negative"]:
                            flag = False

                        speakers.append(speaker["name"])

                for speak in speakers:
                    if speakers.count(speak) > 1:
                        flag = False

                for turn in test["conversation"]:
                    if "message" not in turn.keys() or "id" not in turn.keys():
                        flag = False
                    if turn["speaker"] not in speakers:
                        flag = False
                    for add in turn["addressee"]:
                        if add not in speakers:
                            flag = False


        except:
            flag = False

        if flag:
            good += 1
            good_list.append(1)
        else:
            good_list.append(0)

    print(f"Total number of conversations: {len(datalist)}")
    print(f"Number of conversations with correct format: {good}")

    return good_list

def check_number_of_speakers(datalist, filtering):

    good = 0
    good_list = []

    for idx, data in enumerate(datalist):
        try:
            test = json.loads(data)
            if len(test["speakers"]) == filtering[idx]["total_user"]:
                good += 1
                good_list.append(1)
            else:
                good_list.append(0)
        except:
            good_list.append(0)
            pass

    print(f"Total number of conversations: {len(datalist)}")
    print(f"Number of conversations with correct number of speakers: {good}")

    return good_list

def check_everyone_speaks(datalist):
    good = 0
    good_list = []

    for idx, data in enumerate(datalist):
        test = json.loads(data)

        for sp in test["speakers"]:
            flag = False
            for turn in test["conversation"]:
                if turn["speaker"] == sp["name"]:
                    flag = True
                    break
            if not flag:
                break

        if flag:
            good_list.append(1)
            good += 1
        else:
            good_list.append(0)
        pass

    print(f"Total number of conversations: {len(datalist)}")
    print(f"Number of conversations with everyone speaking: {good}")

    return good_list

def check_stances(datalist, filtering):
    good = 0
    count = 0
    good_list = []
    for idx, data in enumerate(datalist):
        try:
            test = json.loads(data)

            stances = {"pro": 0, "cons":0}


            for sp in test["speakers"]:
                if sp["stance"].lower() == "positive":
                    stances["pro"] += 1
                if sp["stance"].lower() == "negative":
                    stances["cons"] += 1

            if stances["pro"] == (int(filtering[idx]["total_user"]/2) + filtering[idx]["difference"]) and stances["cons"] == (filtering[idx]["total_user"] - int(filtering[idx]["total_user"]/2) - filtering[idx]["difference"]):
                good += 1
                good_list.append(1)
            else:
                good_list.append(0)
        except:
            count += 1
            good_list.append(0)
            pass

    print(f"Total number of conversations: {len(datalist)}")
    print(f"Number of conversations with balanced: {good}")

    return good_list

def check_number_of_turns(datalist, filtering):

    good = 0
    good_list = []

    for idx, data in enumerate(datalist):
        try:
            test = json.loads(data)

            if len(test["conversation"]) == 15:
                good += 1
                good_list.append(1)
            else:
                good_list.append(0)
        except:
            good_list.append(0)
            pass

    print(f"Total number of conversations: {len(datalist)}")
    print(f"Number of conversations with correct number of turns: {good}")

    return good_list
def print_conversation(data):
    test = json.loads(data)

    for turn in test["conversation"]:
        print(f"Speaker: {turn['speaker']}")
        print(f"Addressee: {turn['addressee']}")
        print(f"Utterance: {turn['message']}")
        print("")

def create_graph(data, pro, cons):
    authors_id = dict()
    nodes = []
    edges = []
    id = 0

    test = json.loads(data)

    for s in test["speakers"]:
        authors_id[s["name"]] = id
        id += 1

        nodes.append(authors_id[s["name"]])


    for turn in test["conversation"]:
        for add in turn["addressee"]:
            edges.append((authors_id[turn["speaker"]], authors_id[add]))

    G = nx.MultiDiGraph()

    G.add_nodes_from(nodes)

    G.add_edges_from(edges)
    #assign label to nodes
    for speaker in test["speakers"]:
        key = speaker["name"]
        if speaker["stance"].lower() in pro:
            G.nodes[authors_id[key]]["label"] = "pro"
            G.nodes[authors_id[key]]["color"] = "blue"
        else:
            if speaker["stance"].lower() in cons:
                G.nodes[authors_id[key]]["label"] = "cons"
                G.nodes[authors_id[key]]["color"] = "red"
            else:
                G.nodes[authors_id[key]]["label"] = "other"
                G.nodes[authors_id[key]]["color"] = "grey"
    return G

def create_collection_of_graphs(datalist, pro, cons):
    graphs = []
    for idx, data in enumerate(datalist):
        G = create_graph(data, pro, cons)
        graphs.append(G)
    return graphs


def plot_graph_easy(G):
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=100, font_size = 10)
    plt.draw()
    plt.show()


def plot_graph_complex(G):
    pos = nx.random_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color = 'r', node_size = 100, alpha = 1)
    ax = plt.gca()
    for e in G.edges:
        ax.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])
                                                                           ),
                                    ),
                    )
        #print ids of nodes
        ax.annotate(e[0], pos[e[0]], xytext=(-10,10), textcoords='offset points')
    plt.axis('off')
    plt.show()

def print_degree(G):
    for node in G.nodes:
        print(f"Node {node} has in-degree {G.in_degree(node)} and out-degree {G.out_degree(node)}")

def count_occurence_edges(G):
    edges = dict()
    for edge in G.edges:
        if edge not in edges.keys():
            edges[edge] = 1
        else:
            edges[edge] += 1
    return edges


#from a graph, create a list of all the subgraphs of size 2
def create_subgraphs(G):
    subgraphs = []
    for node in G.nodes:
        for neighbor in G.neighbors(node):
            if node < neighbor:
                subgraphs.append(G.subgraph([node, neighbor]))
    return subgraphs

def create_turn_subgraphs(data, authors_id):
    test = json.loads(data)
    subgraphs = []
    for turn in test["conversation"]:
        edges = []

        for add in turn["addressee"]:
            edges.append((authors_id[turn["speaker"]], authors_id[add]))

        G = nx.MultiDiGraph()
        G.add_edges_from(edges)
        subgraphs.append(G)

    return subgraphs


def check_number_of_first_addr(datalist):

    good = 0
    good_list = []

    for idx, data in enumerate(datalist):
        test = json.loads(data)

        if len(test["conversation"])>0:
            if len(test["conversation"][0]["addressee"]) == (len(test["speakers"])-1):
                good += 1
                good_list.append(1)
            else:
                good_list.append(0)
        else:
            good_list.append(0)


    print(f"Total number of conversations: {len(datalist)}")
    print(f"Number of conversations with correct number of first_addressees: {good}")
    return good_list

def check_turns_greater_than_2(datalist):

    good = 0
    good_list = []

    for idx, data in enumerate(datalist):
        test = json.loads(data)

        if len(test["conversation"])/len(test["speakers"])>=1.9:
            good += 1
            good_list.append(1)
        else:
            good_list.append(0)

    print(f"Total number of conversations: {len(datalist)}")
    print(f"Number of conversations with more than 10 turns: {good}")
    return good_list

def remanaged_conversation(conv):
    new_conv = {"conversation": [], "speakers": conv["speakers"]}

    for turn in conv["conversation"]:
        if "id" in turn.keys():
            if "speaker" in turn.keys() and turn["speaker"] != "" and turn["speaker"] != None:
                if "addressee" in turn.keys() and turn["addressee"] != "" and turn["addressee"] != [None]:
                    if "message" in turn.keys() and turn["message"] != "" and turn["message"] != [None]:
                        new_conv["conversation"].append(turn)

    return new_conv

def check_no_wrong_addressees(datalist):
    good = 0
    good_list = []

    for idx, data in enumerate(datalist):
        test = json.loads(data)

        if len(test["conversation"])>0:
            flag = True
            for turn in test["conversation"]:
                if turn["speaker"] in turn["addressee"] or len(turn["addressee"]) == 0 or turn["addressee"] == [None]:
                    flag = False

            if flag:
                good += 1
                good_list.append(1)
            else:
                good_list.append(0)

        else:
            good_list.append(0)


    print(f"Total number of conversations: {len(datalist)}")
    print(f"Number of conversations with no self loops: {good}")
    return good_list

def full_checks(data, filtering):
    good_format = check_correct_format(data)

    filtering_good = [x for x, y in zip(filtering, good_format) if y]

    data_good = [x for x, y in zip(data, good_format) if y]

    # Determine the bins based on the combined data range
    bins = np.linspace(min(good_format),max(good_format), 3)  # Adjust number of bins as needed

    # Plot each histogram

    plt.hist(good_format, bins=bins, density=True, color='lightgreen', edgecolor='black')

    # Adjust layout
    plt.tight_layout()

    plt.show()

    good_speakers = check_number_of_speakers(data_good, filtering_good)

    # Determine the bins based on the combined data range
    bins = np.linspace(min(good_speakers),max(good_speakers), 3)  # Adjust number of bins as needed


    # Plot each histogram
    plt.hist(good_speakers, bins=bins, density=True, color='lightgreen', edgecolor='black')

    # Adjust layout
    plt.tight_layout()
    plt.show()


    good_turns = check_number_of_turns(data_good, filtering_good)

    # Determine the bins based on the combined data range
    bins = np.linspace(min(good_turns), max(good_turns), 3)  # Adjust number of bins as needed

    # Create subplots

    # Plot each histogram

    plt.hist(good_turns, bins=bins, density=True, color='lightgreen', edgecolor='black')

    # Adjust layout
    plt.tight_layout()
    plt.show()


    good_first_addressees = check_number_of_first_addr(data_good)


    bins = np.linspace(min(good_first_addressees), max(good_first_addressees), 3)  # Adjust number of bins as needed

    plt.hist(good_first_addressees, bins=bins, density=True, color='lightgreen', edgecolor='black')

    plt.tight_layout()
    plt.show()


    good_addr = check_no_wrong_addressees(data_good)

    # Determine the bins based on the combined data range
    bins = np.linspace(min(good_addr), max(good_addr), 3)  # Adjust number of bins as needed

    # Create subplots

    # Plot each histogram

    plt.hist(good_addr, bins=bins, density=True, color='lightgreen', edgecolor='black')

    # Adjust layout
    plt.tight_layout()
    plt.show()

    good_stances = check_stances(data_good, filtering_good)


    # Determine the bins based on the combined data range
    bins = np.linspace(min(good_stances), max(good_stances), 3)  # Adjust number of bins as needed


    plt.hist(good_stances, bins=bins, density=True, color='lightgreen', edgecolor='black')

    # Adjust layout
    plt.tight_layout()
    plt.show()

    good_everyone_speaks = check_everyone_speaks(data_good)

    # Determine the bins based on the combined data range
    bins = np.linspace(min(good_everyone_speaks), max(good_everyone_speaks), 3)  # Adjust number of bins as needed


    plt.hist(good_everyone_speaks, bins=bins, density=True, color='lightgreen', edgecolor='black')

    # Adjust layout
    plt.tight_layout()
    plt.show()


    return good_format, data_good, filtering_good, good_speakers, good_turns, good_first_addressees, good_addr, good_stances, good_everyone_speaks


#%%
