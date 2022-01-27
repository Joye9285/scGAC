# scGAC
single-cell graph attentional clustering

## Requirements

Python --- 3.6.4

Tensorflow --- 1.12.0

Keras --- 2.1.0

Numpy --- 1.19.5

Scipy --- 1.5.4

Pandas --- 1.1.5

Sklearn --- 0.24.2

## Usage
To use scGAC, you should specify the two parameters, `dataset_str` and `n_clusters`, and run the following command:
```Bash
python scGAC.py dataset_str n_clusters
```
where `dataset_str` is the name of dataset and `n_clusters` is the number of clusters.<br>

## Example
For example, for dataset `Yan`, you can run the following command:
```Bash
python scGAC.py Yan 6
```
For your own dataset named `DatasetX`, you can first create a new folder under `\data`, and put the expression matrix file `data.tsv` into `data/Dataset_X/`, then run scGAC on it.<br>

### Inputs

The tested datasets can be downloaded from: ([Yan](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36552), [Biase](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57249), [Klein](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525), [Romanov](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE74672), [Muraro](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85241), [Björklund](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70580), [PBMC](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc6k), [Zhang](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108989), [Guo](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE99254), [Brown.1](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137710), [Brown.2](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137710), [Chung](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75688), [Sun.1](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128066), [Sun.2](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128066), [Sun.3](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128066) and [Habib](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104525)). 

Please note that we recommend you use the `raw count` expression matrix as the input of scGAC. 

### Outputs
You can obtain the predicted clustering result `pred_DatasetX.txt` and the learned cell embeddings `hidden_DatasetX.tsv` under the folder `result/`.

### Optional parameters
To see the optional parameters, you can run the following command:
```Bash
python scGAC.py -h
```
For example, if you want to evaluate the clustering results (by specifing `--subtype_path`) and change the number of nearest neighbors (by specifing `--k`), you can run the following command:
```Bash
python scGAC.py Yan 6 --subtype_path data/Yan/subtype.ann --k 4
```
Results in the paper were obtained with default parameters.
