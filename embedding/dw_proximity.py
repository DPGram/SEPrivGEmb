import networkx as nx
import numpy as np

def construct_matrix_P(G):
    n = len(G.nodes())  # Number of nodes in the graph
    P = np.zeros((n, n))  # Initialize matrix P with zeros

    # Calculate degrees of all nodes
    degrees = dict(G.degree())

    # Construct matrix P
    for i in G.nodes():
        degree_i = degrees[i]
        if degree_i > 0:
            for j in G.neighbors(i):
                P[i][j] = 1 / degree_i

    # Compute P^2
    P2 = np.matmul(P, P)

    # Calculate (P + P^2) / 2
    result = (P + P2) / 2

    return result
