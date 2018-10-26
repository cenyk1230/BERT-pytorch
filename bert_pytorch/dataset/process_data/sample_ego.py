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
    parser.add_argument('--number-walks', type=float, default=10)
    parser.add_argument('--walk-length', type=float, default=20)
    parser.add_argument('--threshold', type=int, default=10000)
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

    egos = []
    for u in nodelist * args.number_walks:
        ego = [u]
        walker = random_walk_with_restart(
            graph, start=[u], restart_prob=args.restart_prob)
        trial = 0
        while len(ego) < args.walk_length:
            v = walker.__next__()
            if v not in ego:
                trial = 0
                ego.append(v)
            else:
                trial += 1
            if trial >= args.threshold:
                break
        egos.append(ego)

    with open(args.edgelist + '.walks', 'w') as f:
        for ego in egos:
            f.write(' '.join([str(v) for v in ego]) + '\n')

