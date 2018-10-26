import graph
import random
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--edgelist', type=str)
    parser.add_argument('--number-walks', type=float, default=10)
    parser.add_argument('--walk-length', type=float, default=20)

    args = parser.parse_args()

    G = graph.load_edgelist(args.edgelist, undirected=True)

    walk_length = args.walk_length
    num_walks = args.number_walks

    walks = graph.build_deepwalk_corpus(G, num_paths=num_walks, path_length=walk_length, alpha=0, rand=random.Random(1995))

    with open(args.edgelist + '.walks', 'w') as out:
        for walk in walks:
            out.write(' '.join(walk) + '\n')

