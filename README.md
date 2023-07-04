# background-clustering
background-clustering(k-means, dbscan, hierarchical clustering) made by prolcy

k-means

```bash
  python main.py -k {cluster_count}
  ```

dbscan
```bash
  python main.py -d {eps} {min_samples}
  ```
hierarchical clustering
```bash
  python main.py -h {cluster_count}
  ```

# visualization
The image, json, csv file should exist.

cluster_viewer

```bash
  python visual_main.py -c
  ```

nearest 5 images

```bash
  python visual_main.py -n
  ```
