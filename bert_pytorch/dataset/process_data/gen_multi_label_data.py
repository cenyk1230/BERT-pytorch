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
    parser.add_argument('-tr', '--train_rate', default=0.8, type=float)
    parser.add_argument('-s', '--seed', default=1230, type=int)

    args = parser.parse_args()

    random.seed(args.seed)

    if args.mode == 0:
        os.system('python sample_ego.py -e %s --number-walks=1' % args.edgelist)
        walks = load_from_file(args.edgelist + '.walks')
    else:
        os.system('python sample_walk.py -e %s -number-walks=1' % args.edgelist)
        walks = load_from_file(args.edgelist + 'walks')

    dire = args.edgelist.split('.')[0] + '_finetune'

    train_rate = args.train_rate
    if args.edgelist == "flickr.edgelist":
        train_rate /= 10

    walks_dic = {}
    for walk in walks:
        walks_dic[int(walk[0])] = walk

    label_num = 0

    labels = {}
    with open(args.label, 'r') as f:
        for line in f:
            items = line.strip().split()
            x = int(items[0])
            l = int(items[1])
            label_num = max(label_num, l)
            if x not in labels:
                labels[x] = {l:1}
            else:
                labels[x][l] = 1

    nodes = list(range(len(walks)))
    train_node, test_node = train_test_split(nodes, train_size=train_rate, random_state=args.seed)
    train_node, valid_node = train_test_split(train_node, train_size=7.0/8, random_state=args.seed)

    with open(dire + '/train', 'w') as out:
        for v in train_node:
            out.write(' '.join(walks_dic[v]) + '\t' + ' '.join([str(1 if x in labels[v] else 0) for x in range(label_num+1)]) + '\n')

    with open(dire + '/valid', 'w') as out:
        for v in valid_node:
            out.write(' '.join(walks_dic[v]) + '\t' + ' '.join([str(1 if x in labels[v] else 0) for x in range(label_num+1)]) + '\n')

    with open(dire + '/test', 'w') as out:
        for v in test_node:
            out.write(' '.join(walks_dic[v]) + '\t' + ' '.join([str(1 if x in labels[v] else 0) for x in range(label_num+1)]) + '\n')


