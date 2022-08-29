# point-dl
Point based deep learning modules and associated tools for classification.

Contents
----
[Models](https://github.com/Brent-Murray/point-dl/tree/main/models)
| Model | Description|
| ----- | -----------|
| [Classifier](https://github.com/Brent-Murray/point-dl/blob/main/models/classifier.py) | Script used for classification using outputs of other models |
  * dgcnn.py - A script with the Dynamic Graph CNN model architecture
  * pointcnn.py - A script with the PointCNN model architecture
  * pointnet2.py - A script with the PointNet++ model architecture
* utils
  * augmentation.py - A script that performs augmentations on point clouds
  * data_prep.py - A script that splits the dataset and outputs a pickel and csv of the filepaths and species composition
  * resample_point_cloud.py - A script that resamples point clouds and writes them out
  * tools.py - A script with useful tools for point cloud deep learning and classification
  * train.py - A script that defines training/validation/testing process
* dgcnn_main.py - A script that runs the Dynamic Graph CNN Model
* pointcnn_main.py - A script that runs the PointCNN Model
* pointnet2_main.py - A script that runs the PointNet++ Model
