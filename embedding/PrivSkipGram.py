import tensorflow as tf
import numpy as np
import argparse
import networkx as nx
from rdp_accountant import compute_rdp, get_privacy_spent
import functions

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', default=128)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
parser.add_argument('--edge_sampling', default='numpy', help='numpy or atlas or uniform')
parser.add_argument('--node_sampling', default='numpy', help='numpy or atlas or uniform')
parser.add_argument('--lr', default=0.1)  # done
parser.add_argument('--k', default=5)
parser.add_argument('--sigma', default=5)
parser.add_argument('--delta', default=10**(-5))
parser.add_argument('--epsilon', default=0.5)
parser.add_argument('--RDP', default=True)
parser.add_argument('--clip_value', default=2)

class skip_gram:
    def __init__(self, num_of_nodes):
        with tf.compat.v1.variable_scope('forward_pass'):
            tf.compat.v1.disable_eager_execution()
            self.u_i = tf.compat.v1.placeholder(name='u_i', dtype=tf.int32, shape=[None])
            self.u_j = tf.compat.v1.placeholder(name='u_j', dtype=tf.int32, shape=[None])
            self.label = tf.compat.v1.placeholder(name='label', dtype=tf.float32, shape=[None])
            self.edge_weight = tf.compat.v1.placeholder(name='edge_sampleProb', dtype=tf.float32, shape=[None])

            self.embedding = tf.compat.v1.get_variable('target_embedding', [num_of_nodes, args.embedding_dim],
                                             initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=num_of_nodes), self.embedding)

            if args.proximity == 'first-order':
                self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)
            elif args.proximity == 'second-order':
                self.context_embedding = tf.compat.v1.get_variable('context_embedding', [num_of_nodes, args.embedding_dim],
                                                         initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
                self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=num_of_nodes), self.context_embedding)

            # ------------------------------------------------------------
            self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)
            self.loss = tf.reduce_mean(-tf.compat.v1.log_sigmoid(self.label * self.inner_product) * self.edge_weight)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr)
            self.params = [v for v in tf.compat.v1.trainable_variables() if 'forward_pass' in v.name]

            if args.RDP:
                self.var_list = self.params
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss, self.var_list)
                for i, (g, v) in enumerate(self.grads_and_vars):
                    if g is not None and v is not None:
                        if 'target_embedding' in v.name:
                            # gradient clipping
                            g = tf.clip_by_norm(g, args.clip_value)
                            # Add noise only to non-zero gradients
                            non_zero_indices = tf.where(tf.not_equal(g, 0))[:, 0]
                            non_zero_g = tf.gather(g, non_zero_indices)
                            total_batch_num = args.batch_size * (args.k + 1)
                            new_sigma = args.sigma * args.batch_size * (args.k + 1)
                            noisy_g = non_zero_g + tf.compat.v1.random_normal(tf.shape(non_zero_g),
                                stddev=new_sigma * args.clip_value / total_batch_num)
                            # Put the noise-added non-zero gradient back into the original gradient
                            g = tf.tensor_scatter_nd_update(g, tf.expand_dims(non_zero_indices, axis=-1), noisy_g)
                            # Update gradient-variable pair
                            self.grads_and_vars[i] = (g, v)
                        else:
                            # gradient clipping
                            g = tf.clip_by_norm(g, args.clip_value)
                            # Update gradient-variable pair
                            self.grads_and_vars[i] = (g, v)

                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)
            else:
                self.train_op = self.optimizer.minimize(self.loss)

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

class edge_sampling:
    def __init__(self, graph_file=None):
        self.g = graph_file
        self.num_of_nodes = len(self.g.nodes())
        self.num_of_edges = len(self.g.edges())
        self.edges_raw = self.g.edges(data=True)
        self.nodes_raw = self.g.nodes(data=True)

        self.edge_distribution = np.array([attr['weight'] for _, _, attr in self.edges_raw], dtype=np.float32)
        self.total_weights = np.sum(self.edge_distribution)
        self.edge_distribution /= self.total_weights
        self.edge_sampling = AliasSampling(prob=self.edge_distribution)
        self.node_negative_distribution = np.power(
            np.array([self.g.degree(node, weight='weight') for node, _ in self.nodes_raw], dtype=np.float32),
            1/self.total_weights)
        self.node_negative_distribution /= np.sum(self.node_negative_distribution)
        self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

        self.node_index = {}
        self.node_index_reversed = {}
        for index, (node, _) in enumerate(self.nodes_raw):
            self.node_index[node] = index
            self.node_index_reversed[index] = node
        self.edges = [(self.node_index[u], self.node_index[v]) for u, v, _ in self.edges_raw]

    def pos_neg_sampling(self):
        global edge_batch_index, negative_node
        if args.edge_sampling == 'numpy':
            # edge_batch_index = np.random.choice(self.num_of_edges, size=args.batch_size, p=self.edge_distribution)
            edge_batch_index = np.random.choice(self.num_of_edges, size=args.batch_size, replace=False)
        elif args.edge_sampling == 'atlas':
            edge_batch_index = self.edge_sampling.sampling(args.batch_size)
        elif args.edge_sampling == 'uniform':
            edge_batch_index = np.random.randint(0, self.num_of_edges, size=args.batch_size)

        u_i = []
        u_j = []
        label = []
        edge_weight = []

        count = 0
        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]
            if self.g.__class__ == nx.Graph:
                if np.random.rand() > 0.5:
                    edge = (edge[1], edge[0])

            u_i.append(edge[0])
            u_j.append(edge[1])
            label.append(1)
            edge_weight.append(1/self.g.degree(edge[0]))
            count = count + 1

            for i in range(args.k):
                while True:
                    if args.node_sampling == 'numpy':
                        negative_node = np.random.choice(self.num_of_nodes, p=self.node_negative_distribution)
                    elif args.node_sampling == 'atlas':
                        negative_node = self.node_sampling.sampling()
                    elif args.node_sampling == 'uniform':
                        negative_node = np.random.randint(0, self.num_of_nodes)
                    if not self.g.has_edge(self.node_index_reversed[negative_node], self.node_index_reversed[edge[0]]):
                        break
                u_i.append(edge[0])
                u_j.append(negative_node)
                label.append(-1)
                edge_weight.append(1/self.g.degree(edge[0]))

        return u_i, u_j, label, edge_weight

class trainModel:
    def __init__(self, graph):
        self.graph = graph
        self.num_of_nodes = graph.number_of_nodes()
        self.model = skip_gram(self.num_of_nodes)
        self.edge_sampling = edge_sampling(self.graph)

    def train(self, test_task=None):
        with tf.compat.v1.Session() as sess:
            print(args)
            print('batches\tloss\tsampling time\ttraining_time\tdatetime')
            # if d_epoch == 0:
            sess.run(tf.compat.v1.global_variables_initializer())

            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
            # orders = np.arange(2, 32, 0.1)
            rdp = np.zeros_like(orders, dtype=float)
            for each_epoch in range(args.n_epoch):
                u_i, u_j, label, edge_weight = self.edge_sampling.pos_neg_sampling()
                feed_dict = {self.model.u_i: u_i, self.model.u_j: u_j, self.model.label: label,
                             self.model.edge_weight: edge_weight}
                _, loss = sess.run([self.model.train_op, self.model.loss], feed_dict=feed_dict)

                # RDP mechanism
                number_of_edges = len(self.graph.edges())
                sampling_prob = args.batch_size / number_of_edges
                steps = each_epoch + 1
                new_sigma = args.sigma * args.batch_size * (args.k + 1)
                rdp = compute_rdp(q=sampling_prob, noise_multiplier=new_sigma, steps=steps, orders=orders)
                rdp = rdp + compute_rdp(q=1, noise_multiplier=new_sigma, steps=steps, orders=orders)
                _eps, _delta, _ = get_privacy_spent(orders, rdp, target_eps=args.epsilon)
                if _delta > args.delta:
                    print('jump out')
                    break

                # Get node embeddings
                embedding = sess.run(self.model.embedding)

                if test_task == 'StrucEqu':
                    A = nx.to_numpy_matrix(trainGraph)
                    A = np.array(A)
                    pearson_vals = functions.structural_equivalence(A, embedding)
                    pearson_val = pearson_vals[0]
                    print(each_epoch, pearson_val)

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

if __name__ == '__main__':
    test_task = 'StrucEqu'  # input StrucEqu or lp
    set_algo_name = 'PrivSkipGram'
    parser.add_argument('--n_epoch', default=200)  # all datasets are set to 5
    args = parser.parse_args()  # parameter

    oriGraph_filename = '../data/PPI/train_1'

    # Load graph
    trainGraph = loadGraphFromEdgeListTxt(oriGraph_filename, directed=False)

    print('Num nodes: %d, num edges: %d' % (trainGraph.number_of_nodes(), trainGraph.number_of_edges()))
    tm = trainModel(trainGraph)
    tm.train(test_task=test_task)
