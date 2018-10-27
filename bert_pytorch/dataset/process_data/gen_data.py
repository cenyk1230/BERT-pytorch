import os
import graph
import random
import argparse

from sklearn.model_selection import train_test_split


def load_from_file(walk_file):
    walks = []
    with open(walk_file, 'r') as f:
        for line in f:
            walks.append(line.strip().split())
    return walks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--edgelist', required=True, type=str)
    parser.add_argument('-l', '--label', required=True, type=str)
    parser.add_argument('-m', '--mode', default=0, type=int)

    args = parser.parse_args()

    if args.mode == 0:
        os.system('python sample_ego.py -e %s --number-walks=1' % args.edgelist)
        walks = load_from_file(args.edgelist + '.walks')
    else:
        os.system('python sample_walk.py -e %s -number-walks=1' % args.edgelist)
        walks = load_from_file(args.edgelist + 'walks')

    dire = 'blog_data'
    if args.edgelist != 'blogcatalog.edgelist':
        dire = args.edgelist.split('.')[0] + '_data'

    walks_dic = {}
    for walk in walks:
        walks_dic[int(walk[0])] = walk

    labels = {}
    with open(args.label, 'r') as f:
        for line in f:
            items = line.strip().split()
            x = int(items[0])
            l = int(items[1])
            if l not in labels:
                labels[l] = [x]
            else:
                labels[l].append(x)

    all_nodes = set(range(len(walks)))

    for i in labels.keys():
        nodes = set(labels[i])
        neg_nodes = list(all_nodes - set(nodes))
        nodes = list(nodes)

        train_node1, test_node1 = train_test_split(nodes, train_size=0.8)
        train_node2, test_node2 = train_test_split(neg_nodes, train_size=0.8)

        with open(dire + '/train%d' % i, 'w') as out:
            for v in train_node1:
                out.write(' '.join(walks_dic[v]) + '\t1\n')
            for v in train_node2:
                out.write(' '.join(walks_dic[v]) + '\t0\n')

        with open(dire + '/test%d' % i, 'w') as out:
            for v in test_node1:
                out.write(' '.join(walks_dic[v]) + '\t1\n')
            for v in test_node2:
                out.write(' '.join(walks_dic[v]) + '\t0\n')


