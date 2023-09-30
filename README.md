# Person Group Trajectory Dataset



## Description
Person Group Trajectory Dataset(PGTD) comes from paper `People Group Detection with Global Trajectory Extraction in a Disjoint Camera Network`. This is an expanded version of the dataset PTD, in which we added group information and reorganized the dataset.

Fig. 1:The spatial distribution of the cameras in the Person Group Trajectory Dataset. For each camera, the satellite enlarged image and the camera view of the corresponding cameras are displayed. Different colored line segments represent different single-camera tracklets.

![Figure1](https://github.com/zhangxin1995/PTD_GROUP/blob/8a3827420bc3a10856604f08a98b3e314f590517/images/location.png)

Fig. 2:(A) (B) (C) (D) and (E) represent the trajectories extracted by the CCRF. The time below the picture indicates the earliest occurrence of the corresponding track.

![Figure2](https://github.com/zhangxin1995/PTD_GROUP/blob/8a3827420bc3a10856604f08a98b3e314f590517/images/example.png)

Fig. 3: The diagram of the group detection framework, the trajectory retrieval process, and the cyclic conditional random field method (CCRF) proposed in this paper.
![Figure3](https://github.com/zhangxin1995/PTD_GROUP/blob/8a3827420bc3a10856604f08a98b3e314f590517/images/framework.png)

## Person Group Trajectory Dataset

You can load the dataset through the following code:
```python
import pickle as pkl
with open('resnet50_visual_dataset.pkl','rb') as infile:
    datas=pkl.load(infile)
```
Before introducing the format of the visual dataset, we should first understand several indexes in the process of person retrieval. In the dataset, we have 5-class indexes, which are as follows:
1. Person index. It represents the identity of pedestrians. Everyone has a unique index.
2. Camera index. It represents the camera number, and each camera has a unique index.
3. Camera tracklet index. It represents the index of a tracklet under a specific camera.
4. Global tracklet index. The global index is obtained by splicing the tracklets under all cameras together.
5. Trajectory index. The trajectory index indicates a tracklet belongs to which trajectory.

The dataset are represented by a `dict`, and each of them has the following meanings:
1. qcs: Camera index list of query tracklets.
2. qts: Timestamp list of query tracklets.
3. qfs: Feature list list of query tracklets. 
4. qls: Person index list of query tracklets.
5. fqcs: Camera list index list of query images.
6. tidxs: Camera tracklet index list of gallery tracklets.
7. fqfs: Feature list of query image.
8. fqls: Person index list of query images.
9. tcs: Camera index list of gallery tracklets.
10. gts: Timestamp list of gallery tracklets.
11. tfs: Feature list of  gallery tracklets.
12. tls: Person index list of gallery tracklets.
13. ftcs: Camera index list of gallery images.
14. ftfs: Feature list of gallery images.
15. ftls: Person index list of gallery images.
16. ftidxs: Camera tracklet index list of gallery images.
17. idx2pathidx: Map the person index to the global tracklet index.
18. tpath2index:Map the trajectory index to the global tracklet index.
19. qidxs: Camera tracklet index list of query tracklets.
20. fqidxs: Camera tracklet index list of query images.
23. group_pidxs_dict: Group information for each ID.


## Code
Please put the downloaded code file and execute the following statement:
```
git clone https://github.com/zhangxin1995/PTD_GROUP
cd code
python main.py --yaml ./config/global_crf_resnet.yaml
```















