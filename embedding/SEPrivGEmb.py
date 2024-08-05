import random
import tensorflow as tf
import numpy as np
import argparse
import networkx as nx
import yaml
from rdp_accountant import compute_rdp, get_privacy_spent
import functions

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', default=128)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
parser.add_argument('--edge_sampling', default='numpy', help='numpy or atlas or uniform')
parser.add_argument('--node_sampling', default='numpy', help='numpy or atlas or uniform')
parser.add_argument('--lr', default=0.1)
parser.add_argument('--k', default=5)
parser.add_argument('--sigma', default=10)
parser.add_argument('--delta', default=10**(-5))
parser.add_argument('--epsilon', default=3)
parser.add_argument('--RDP', default=True)
parser.add_argument('--clip_value', default=2)
parser.add_argument('--n_epoch', default=500)

class PrivGEmb:
    def __init__(self, num_of_nodes):
        with tf.compat.v1.variable_scope('forward_pass'):
            tf.compat.v1.disable_eager_execution()
            self.u_i = tf.compat.v1.placeholder(name='u_i', dtype=tf.int32, shape=[None])
            self.u_j = tf.compat.v1.placeholder(name='u_j', dtype=tf.int32, shape=[None])
            self.label = tf.compat.v1.placeholder(name='label', dtype=tf.float32, shape=[None])
            self.prox_val = tf.compat.v1.placeholder(name='edge_sampleProb', dtype=tf.float32, shape=[None])

            self.embedding = tf.compat.v1.get_variable('target_embedding', [num_of_nodes, args.embedding_dim],
                                             initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=num_of_nodes), self.embedding)

            if args.proximity == 'first-order':
                self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)
            elif args.proximity == 'second-order':
                self.context_embedding = tf.compat.v1.get_variable('context_embedding', [num_of_nodes, args.embedding_dim],
                                                         initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
                self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=num_of_nodes), self.context_embedding)

            self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)
            self.loss = tf.reduce_mean(-tf.compat.v1.log_sigmoid(self.label * self.inner_product) * self.prox_val)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr)
            self.params = [v for v in tf.compat.v1.trainable_variables() if 'forward_pass' in v.name]

            if args.RDP:
                self.var_list = self.params
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss, self.var_list)
                for i, (g, v) in enumerate(self.grads_and_vars):
                    g = tf.clip_by_norm(g, args.clip_value)
                    non_zero_indices = tf.where(tf.not_equal(g, 0))[:, 0]
                    non_zero_g = tf.gather(g, non_zero_indices)
                    noisy_g = non_zero_g + tf.compat.v1.random_normal(tf.shape(non_zero_g),
                              stddev=args.sigma * args.clip_value)
                    g = tf.tensor_scatter_nd_update(g, tf.expand_dims(non_zero_indices, axis=-1), noisy_g)
                    self.grads_and_vars[i] = (g, v)

                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)
            else:
                self.train_op = self.optimizer.minimize(self.loss)

class trainModel:
    def __init__(self, graph):
        self.graph = graph
        self.num_of_nodes = graph.number_of_nodes()
        self.model = PrivGEmb(self.num_of_nodes)

    def train(self, subgra_set):
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            # orders for RDP
            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
            rdp = np.zeros_like(orders, dtype=float)
            for each_epoch in range(args.n_epoch):
                u_i, u_j, label, prox_val = get_batchSample_from_subGraSet(subgra_set, args.batch_size)
                feed_dict = {self.model.u_i: u_i, self.model.u_j: u_j, self.model.label: label,
                             self.model.prox_val: prox_val}
                _, loss = sess.run([self.model.train_op, self.model.loss], feed_dict=feed_dict)

                # --------- RDP mechanism -------------
                number_of_edges = len(self.graph.edges())
                sampling_prob = args.batch_size / number_of_edges
                steps = each_epoch + 1
                rdp = compute_rdp(q=sampling_prob, noise_multiplier=args.sigma, steps=steps, orders=orders)
                _eps, _delta, _ = get_privacy_spent(orders, rdp, target_eps=args.epsilon)
                # print(_eps, _delta)

                if _delta > args.delta:
                    print('jump out')
                    break
                # --------------------------------------
                embedding = sess.run(self.model.embedding)

                A = nx.to_numpy_matrix(trainGraph)
                A = np.array(A)
                pearson_vals = functions.structural_equivalence(A, embedding)
                pearson_val = pearson_vals[0]

                print(each_epoch, pearson_val)

def get_batchSample_from_subGraSet(subgra_set, batch_size):
    ui_uj_label_dict = subgra_set
    ui_uj_label_dict.keys()
    sampled_keys = random.sample(ui_uj_label_dict.keys(), batch_size)

    u_i = []
    u_j = []
    label = []
    prox_val = []
    for key in sampled_keys:
        ui_uj_label_list = ui_uj_label_dict[key]
        for index in ui_uj_label_list:
            # print(index)
            u_i.append(index[0])
            u_j.append(index[1])
            label.append(index[2])
            prox_val.append(index[3])

    return u_i, u_j, label, prox_val

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
    args = parser.parse_args()  # parameter
    set_dataset_name = 'PPI'

    graph_url = '../data/' + set_dataset_name + '/train_1'

    # Load graph
    trainGraph = loadGraphFromEdgeListTxt(graph_url, directed=False)

    print('Num nodes: %d, num edges: %d' % (trainGraph.number_of_nodes(), trainGraph.number_of_edges()))
    tm = trainModel(trainGraph)

    load_SubGraSet = set_dataset_name + '_SubGraSet_batch128' + '.yaml'

    with open(load_SubGraSet, 'r') as f:
        subgra_set = yaml.load(f, Loader=yaml.Loader)

    tm.train(subgra_set)
