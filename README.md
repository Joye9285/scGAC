# scGAC
single-cell graph attentional clustering

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

### Notes
We recommend you use the `raw count` expression matrix as the input of scGAC. 

### Outputs
You can obtain the predicted clustering result `pred_DatasetX.txt` and the learned cell embeddings `hidden_DatasetX.tsv` under the folder `result/`.

### Optional parameters
For example, if you want to evaluate the clustering results (by specifing `--subtype_path`) and change the number of nearest neighbors (by specifing `--k`), you can run the following command:
```Bash
python scGAC.py Yan 6 --subtype_path data/Yan/subtype.ann --k 4
```