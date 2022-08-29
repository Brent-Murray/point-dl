# point-dl
Point based deep learning modules and associated tools for classification.

Contents
----
### Models
| Model | Description|
| ----- | -----------|
| [Classifier](https://github.com/Brent-Murray/point-dl/blob/main/models/classifier.py) | A script used for classification using outputs of other models |
| [DGCNN](https://github.com/Brent-Murray/point-dl/blob/main/models/dgcnn.py) | Dynamic Graph CNN model architecture | 
| [Dual Model](https://github.com/Brent-Murray/point-dl/blob/main/models/dual_model.py) | A script used to combine the outputs of two models based on defined method |
| [PointCNN](https://github.com/Brent-Murray/point-dl/blob/main/models/pointcnn.py) | PointCNN model architecture |
| [PointNet++](https://github.com/Brent-Murray/point-dl/blob/main/models/pointnet2.py) | PointNet++ model architecture |

* utils
  * augmentation.py - A script that performs augmentations on point clouds
  * data_prep.py - A script that splits the dataset and outputs a pickel and csv of the filepaths and species composition
  * resample_point_cloud.py - A script that resamples point clouds and writes them out
  * tools.py - A script with useful tools for point cloud deep learning and classification
  * train.py - A script that defines training/validation/testing process
* dgcnn_main.py - A script that runs the Dynamic Graph CNN Model
* pointcnn_main.py - A script that runs the PointCNN Model
* pointnet2_main.py - A script that runs the PointNet++ Model
