import pprint
import cvxpy as cp
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as pl
import scipy.sparse as sps
import seaborn as sns
import math
import sys
import math
import itertools
import networkx as nx
        
class Graph():

    def __init__(self, matrix, thresh = 5e-5): #assumes concentration matrix

        self.neighbours = {}
        self.edges = {}
        self.node_vals = {}
        self.matrix = matrix
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                
                v = matrix[i,j]

                if v < -thresh or v > thresh:
                    
                    if i not in self.neighbours:
                        self.neighbours[i] = set()

                    self.neighbours[i].add( j )

                    if j not in self.neighbours:
                        self.neighbours[j] = set()

                    self.neighbours[j].add( i )

                    self.edges[(i,j)] = v
                    self.edges[(j,i)] = v

    def get_neigh(self, idx):
        return self.neighbours[idx]

    def get_edge(self, idxa, idxb):
        return self.edges[idxa, idxb]

    def info(self):

        pp = pprint.PrettyPrinter(indent=4)
        
        print("node neighbours: ")
        pp.pprint(self.neighbours)
        print("edges: ")
        pp.pprint(self.edges)

    def plot(self):

        G = nx.DiGraph()

        thresh = 0.5e-5

        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                v = self.matrix[i,j]
                G.add_edge(i, j, weight=v, alpha=0.5)
                
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0e-5]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0e-5]

        pos = nx.spring_layout(G)  # positions for all nodes
        
        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=700)

        edge_alpha_list = [ abs(self.matrix[e[0],e[1]]) for e in elarge ]

        alpha_max = np.amax(edge_alpha_list)
        
        # edges
        edges = nx.draw_networkx_edges(G, pos,
                                       edgelist=elarge,
                                       width=6,
                                       edge_color='g',
                                       arrowstyle='-')

        for i, arc in enumerate(edges):
            arc.set_alpha(edge_alpha_list[i]/alpha_max)

        edge_alpha_list2 = [ abs(self.matrix[e[0],e[1]]) for e in esmall ]

        if len(edge_alpha_list2)>0:
            alpha_max = np.amax(edge_alpha_list2)
    
            edges = nx.draw_networkx_edges(G, pos,
                                           edgelist=esmall,
                                           width=6,
                                           edge_color="r",
                                           style="dashed",
                                           arrowstyle='-')

            for i, arc in enumerate(edges):
                arc.set_alpha(edge_alpha_list2[i]/alpha_max)
        
        # labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

        pl.axis("off")
        # pl.show()

    def set_node_vals(self, idx, val):
        self.node_vals[idx] =  val

    def get_node_vals(self, idx):
        return self.node_vals[idx]
        
    def optimize(self):
        
        pass
    
        # todo: implement belief propagation
        # eg: see section 2.3 of Gaussian Belief Propagation: Theory and Application by Danny Bickson
       
        # stop = False
        
        # while not stop:
            
            
