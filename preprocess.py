import pandas as pd
import numpy as np
import sys

dataset = sys.argv[1]

# Load data
data_path = 'data/{}/ori_data.tsv'.format(dataset)
save_path = 'data/{}/data.tsv'.format(dataset)
# Original shape: [n_gene, n_cell]
data = pd.read_csv(data_path, index_col=0, sep='\t')
cells = data.columns.values
genes = data.index.values
# After transformation: [n_cell, n_gene]
data = data.values.T
print('Data loaded')
print('Before filtering...')
print(' Number of genes is {}'.format(len(genes)))
print(' Number of cells is {}'.format(len(cells)))


# Filter low-quality cells
nGene = []
for i in range(len(data)):
    nGene.append(len(np.argwhere(data[i] > 0)))
nGene = np.array(nGene)
Q1 = np.percentile(nGene, 25)
Q3 = np.percentile(nGene, 75)
IQR = Q3 - Q1
high_v = Q3 + 3 * IQR
low_v = Q1 - 3 * IQR
x = np.argwhere(nGene <= high_v)
y = np.argwhere(nGene >= low_v)
index1 = np.intersect1d(x, y)

nUMI = np.sum(data, axis=1)
Q1 = np.percentile(nUMI, 25)
Q3 = np.percentile(nUMI, 75)
IQR = Q3 - Q1
high_v = Q3 + 3 * IQR
low_v = Q1 - 3 * IQR
x = np.argwhere(nUMI <= high_v)
y = np.argwhere(nUMI >= low_v)
index2 = np.intersect1d(x, y)

index = []
for i in range(len(index1)):
    if index1[i] in index2:
        index.append(index1[i])
index = np.array(index).reshape(-1)

data = data[index]
cells = cells[index]
data = data.T


# Filter low-quality genes
nCell = []
for i in range(len(data)):
    nCell.append(len(np.argwhere(data[i] > 0)))
nCell = np.array(nCell)
index = np.argwhere(nCell >= 3)
index = index.reshape(len(index))

data = data[index]
genes = genes[index]

print('Pre-processing finished')
print('After filtering...')
print(' Number of genes is {}'.format(len(genes)))
print(' Number of cells is {}'.format(len(cells)))


# Save data
data = pd.DataFrame(data, index=genes, columns=cells)
data.to_csv(save_path, sep='\t')
