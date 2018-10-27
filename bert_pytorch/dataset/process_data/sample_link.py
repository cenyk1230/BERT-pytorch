import random
import itertools
import igraph
import argparse
import numpy as np


def random_walk_with_restart(g, start, restart_prob):
    current = random.choice(start)
    stop = False
    while not stop:
        stop = yield current
        current = random.choice(start) if random.random() < restart_prob or g.degree(current) == 0 \
            else random.choice(g.neighbors(current))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--edgelist', type=str)
    parser.add_argument('--restart-prob', type=float, default=0.8)
    parser.add_argument('--number-walks', type=int, default=1)
    parser.add_argument('--walk-length', type=int, default=10)
    parser.add_argument('--threshold', type=int, default=1000)
    args = parser.parse_args()

    with open(args.edgelist, 'r') as f:
        lines = f.read().strip().split('\n')
        edgelist = [[int(v) for v in line.split()] for line in lines]
    nodelist = []
    for edge in edgelist:
        nodelist.append(edge[0])
        nodelist.append(edge[1])
    nodelist = list(set(nodelist))
    # The input graph has node id starting from 0
    assert nodelist == list(range(len(nodelist)))

    graph = igraph.Graph(len(nodelist), directed=False)
    graph.add_edges(edgelist)
    graph.to_undirected()
    graph.simplify(multiple=True, loops=True)

    links = []
    for u, v in edgelist * args.number_walks:

        ego_u = [u]
        walker = random_walk_with_restart(
            graph, start=[u], restart_prob=args.restart_prob)
        trial = 0
        while len(ego_u) < args.walk_length:
            tmp = walker.__next__()
            if tmp not in ego_u:
                trial = 0
                ego_u.append(tmp)
            else:
                trial += 1
            if trial >= args.threshold:
                break

        ego_v = [v]
        walker = random_walk_with_restart(
            graph, start=[v], restart_prob=args.restart_prob)
        trial = 0
        while len(ego_v) < args.walk_length:
            tmp = walker.__next__()
            if tmp not in ego_v:
                trial = 0
                ego_v.append(tmp)
            else:
                trial += 1
            if trial >= args.threshold:
                break

        links.append((ego_u, ego_v))

    with open(args.edgelist + '.links', 'w') as f:
        for link in links:
            f.write(' '.join([str(v) for v in link[0]]) + '\t' + ' '.join([str(v) for v in link[1]]) + '\n')

