import numpy as np
import colorama
import networkx as nx
import matplotlib.pyplot as plt
import scipy

def create_sparse_matrix(node, path_is):
    sparse_mat = scipy.sparse.dok_matrix((node, node), dtype=np.float32)
    number_connected_all = np.zeros(node)
    with open(path_is, 'r') as f:
        for line in f:
            node1, node2 = list(map(int, line.split()))
            number_connected_all[node1-1] = number_connected_all[node1-1]+1
            sparse_mat[node1-1, node2-1] = 1

    with open(path_is, 'r') as f:
        for line in f:
            node1, node2 = list(map(int, line.split()))
            sparse_mat[node1-1, node2-1] = sparse_mat[node1-1, node2-1]/number_connected_all[node1-1]

    return sparse_mat.tocsr()


def pagerank(node, path_is, plot_show, sparse):

    if sparse:
        adjacency_matrix = create_sparse_matrix(node, path_is)
    else:
        adjacency_matrix = np.zeros((node, node))

        with open(path_is, "r") as f:
            for line in f:
                node1, node2 = list(map(int, line.split()))
                adjacency_matrix[node1-1][node2-1] = 1

        if plot_show == 'Y' and ~sparse:
            directedgraph = nx.DiGraph()
            for i in range(node):
                for j in range(node):
                    if adjacency_matrix[i][j] == 1:
                        directedgraph.add_edge(i, j)

            nx.draw(directedgraph, with_labels=True)
            plt.show()

            for i in range(node):
                number_connected = 0
                for j in range(node):
                    if adjacency_matrix[j][i] == 1:
                        number_connected += 1
                for j in range(node):
                    adjacency_matrix[j][i] /= number_connected

    # Counting the number of nodes in each column and storing them in other matrix lets
    # To remove the dangling node we need to dump with a dumping factor

    # Damping
    damping_factor = 0.85
    ranks = [1/node] * node
    iterations = 0
    while True:
        set_ranks = damping_factor * (adjacency_matrix @ ranks) + (1-damping_factor)/node
        iterations += 1
        if np.allclose(ranks, set_ranks):
            break
        ranks = set_ranks

    print()
    print(f"Ranks are as follows : {colorama.Fore.CYAN}")
    print(ranks)


print("-------------------------------------------------------------------------")
print(f"{colorama.Fore.GREEN}\t\t\tWelcome to page rank generator{colorama.Fore.RESET}")
print()
nodes = int(input(f"{colorama.Fore.BLUE}Enter the number of nodes                 : {colorama.Fore.RESET}"))
path = input(f"{colorama.Fore.BLUE}Enter the path of the text file           : {colorama.Fore.RESET}")
print()
if nodes <= 10000:
    graph = input(f"{colorama.Fore.BLUE}Enter 'Y' to view the directed graph or any other character to not : {colorama.Fore.RED}")
else:
    graph = 'N'
if nodes > 10000:
    sparse_matrix = True
else:
    sparse_matrix = False
pagerank(nodes, path, graph, sparse_matrix)
print()
print(f"{colorama.Fore.RESET}-------------------------------------------------------------------------")
