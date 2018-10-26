import scipy.io

mat_names = ['blogcatalog', 'ppi', 'wiki', 'flickr']


for mat_name in mat_names:
    print mat_name
    d = scipy.io.loadmat(mat_name + '.mat')
    adj = d['network']
    rows, cols = adj.nonzero()
    f = open(mat_name + '.edgelist', 'w')
    for i, j in zip(rows, cols):
        f.write('%d %d\n' % (i, j))
    f.close()
    labels = d['group']
    rows, cols = labels.nonzero()
    f = open(mat_name + '.label', 'w')
    for i, j in zip(rows, cols):
        f.write('%d %d\n' % (i, j))
    f.close()
