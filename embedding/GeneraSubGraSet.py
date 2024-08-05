import networkx as nx
import argparse
import numpy as np
import yaml
import dw_proximity

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128)
parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
parser.add_argument('--edge_sampling', default='uniform', help='numpy or atlas or uniform')
parser.add_argument('--node_sampling', default='numpy', help='numpy or atlas or uniform')
parser.add_argument('--K', default=5)

args = parser.parse_args()
class AliasSampling:
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res

class prepare_data:
    def __init__(self, graph_file=None):
        self.g = graph_file
        self.num_of_nodes = len(self.g.nodes())
        self.num_of_edges = len(self.g.edges())
        self.edges_raw = self.g.edges(data=True)
        self.nodes_raw = self.g.nodes(data=True)

        self.edge_distribution = np.array([attr['weight'] for _, _, attr in self.edges_raw], dtype=np.float32)
        self.edge_distribution /= np.sum(self.edge_distribution)
        self.edge_sampling = AliasSampling(prob=self.edge_distribution)
        self.node_negative_distribution = np.power(
            np.array([self.g.degree(node, weight='weight') for node, _ in self.nodes_raw], dtype=np.float32), 0.75)
        self.node_negative_distribution /= np.sum(self.node_negative_distribution)
        self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

        self.node_index = {}
        self.node_index_reversed = {}
        for index, (node, _) in enumerate(self.nodes_raw):
            self.node_index[node] = index
            self.node_index_reversed[index] = node
        self.edges = [(self.node_index[u], self.node_index[v]) for u, v, _ in self.edges_raw]

    def prepare_data_for_dis(self, batch_edges, similarity_scores):
        global edge_batch_index, negative_node
        #
        # if args.edge_sampling == 'numpy':
        #     edge_batch_index = np.random.choice(self.num_of_edges, size=args.batch_size, p=self.edge_distribution)
        # elif args.edge_sampling == 'atlas':
        #     edge_batch_index = self.edge_sampling.sampling(args.batch_size)
        # elif args.edge_sampling == 'uniform':
        #     edge_batch_index = np.random.randint(0, self.num_of_edges, size=args.batch_size)
        node_score_list = np.sum(similarity_scores, axis=1)
        u_i = []
        u_j = []
        label = []
        edge_weight = []

        # for edge_index in edge_batch_index:
        #     edge = self.edges[edge_index]
        # for edge in self.edges:

        batch_end_nodes = [edge[1] for edge in batch_edges]

        for edge in batch_edges:
            sampling_prob = 0
            for end in batch_end_nodes:
                sampling_prob += similarity_scores[edge[0]][end]
            # sampling probability is np.min(similarity_scores)/sampling_prob
            node_negative_distribution = np.power(node_score_list, np.min(similarity_scores) / sampling_prob)
            node_negative_distribution /= np.sum(node_negative_distribution)

            if self.g.__class__ == nx.Graph:
                if np.random.rand() > 0.5:
                    edge = (edge[1], edge[0])
            u_i.append(edge[0])
            u_j.append(edge[1])
            label.append(1)
            edge_weight.append(similarity_scores[edge[0]][edge[1]])  # new addition

            for i in range(args.K):
                while True:
                    if args.node_sampling == 'numpy':
                        negative_node = np.random.choice(self.num_of_nodes, p=self.node_negative_distribution)
                    elif args.node_sampling == 'atlas':
                        negative_node = self.node_sampling.sampling()
                    elif args.node_sampling == 'uniform':
                        negative_node = np.random.randint(0, self.num_of_nodes)
                    if not self.g.has_edge(edge[0], negative_node):
                        break

                u_i.append(edge[0])
                u_j.append(negative_node)
                label.append(-1)
                edge_weight.append(similarity_scores[edge[0]][negative_node])

        return u_i, u_j, label, edge_weight

def loadGraphFromEdgeListTxt(file_name, directed=True):
    with open(file_name, 'r') as f:
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        for line in f:
            edge = line.strip().split()
            if len(edge) == 3:
                w = float(edge[2])
            else:
                w = 1.0
            G.add_edge(int(edge[0]), int(edge[1]), weight=w)
    return G

def graph_to_subgraph_set_multEdgesWithNeg(graph, similarity_scores, min_positive_value):
    ui_uj_label_dict = {}
    edges = graph.edges()
    num_of_nodes = len(graph.nodes())

    node_score_list = np.sum(similarity_scores, axis=1)
    node_negative_distribution = min_positive_value / node_score_list
    node_negative_distribution /= np.sum(node_negative_distribution)

    index = 0
    for each_edge in edges:
        ui_uj_label_proxVal_list = []

        label = 1
        node_prox_val = similarity_scores[each_edge[0]][each_edge[1]]
        ui_uj_label_proxVal_list.append((each_edge[0], each_edge[1], label, node_prox_val))

        for i in range(args.K):
            while True:
                if args.node_sampling == 'numpy':
                    negative_node = np.random.choice(num_of_nodes, p=node_negative_distribution)
                # elif args.node_sampling == 'atlas':
                #     negative_node = self.node_sampling.sampling()
                # elif args.node_sampling == 'uniform':
                #     negative_node = np.random.randint(0, self.num_of_nodes)
                # if not self.g.has_edge(self.node_index_reversed[negative_node], self.node_index_reversed[edge[0]]):
                #     break
                if not graph.has_edge(each_edge[0], negative_node):
                    break

            label = -1
            ui_uj_label_proxVal_list.append((each_edge[0], negative_node, label, node_prox_val))

        ui_uj_label_dict[index] = ui_uj_label_proxVal_list
        index = index + 1

    return ui_uj_label_dict

if __name__ == '__main__':
    prox_name = 'DW_prox'
    dataset_names = ['PPI']
    for set_dataset_name in dataset_names:
        set_split_name = 'train0.9_test0.1'
        batch_num = 128

        oriGraph_filename = '../data/' + set_dataset_name + '/train_1'
        train_filename = '../data/' + set_dataset_name + '/' + set_split_name + '/'

        # Load graph
        Graph = loadGraphFromEdgeListTxt(oriGraph_filename, directed=False)

        if prox_name == 'DW_prox':
            dw_Mat = dw_proximity.construct_matrix_P(Graph)
            # Find all elements greater than 0
            positive_values = dw_Mat[dw_Mat > 0]

            # Find the smallest value greater than 0
            if len(positive_values) > 0:
                min_positive_value = np.min(positive_values)
                print("The smallest value greater than 0 is:", min_positive_value)
            else:
                print("There are no values greater than 0 in the matrix")

            # save node proximity
            dw_name = set_dataset_name + '_dwSimMat' + '_batch' + str(batch_num)
            np.save(dw_name, dw_Mat)

            # data_loader = prepare_data(Graph)
            subgra_set = graph_to_subgraph_set_multEdgesWithNeg(Graph, dw_Mat, min_positive_value)

            # save dict to YAML file
            save_name = set_dataset_name + '_SubGraSet' + '_batch' + str(batch_num) + '.yaml'
            with open(save_name, 'w') as f:
                yaml.dump(subgra_set, f)
