# point-dl
Point based deep learning modules and associated tools for classification.

Contents
----
### Models
| Model | Description |
| ----- | ----------- |
| [Classifier](https://github.com/Brent-Murray/point-dl/blob/main/models/classifier.py) | A script used for classification using outputs of other models |
| [DGCNN](https://github.com/Brent-Murray/point-dl/blob/main/models/dgcnn.py) | Dynamic Graph CNN model architecture | 
| [Dual Model](https://github.com/Brent-Murray/point-dl/blob/main/models/dual_model.py) | A script used to combine the outputs of two models based on defined method |
| [PointCNN](https://github.com/Brent-Murray/point-dl/blob/main/models/pointcnn.py) | PointCNN model architecture |
| [PointNet++](https://github.com/Brent-Murray/point-dl/blob/main/models/pointnet2.py) | PointNet++ model architecture |

### Utils
| Util | Description |
| ---- | ----------- |
| [Augmentation](https://github.com/Brent-Murray/point-dl/blob/main/utils/augmentation.py) | A script that performes augmentations on point clouds |
| [Create Labels](https://github.com/Brent-Murray/point-dl/blob/main/utils/create_labels.py) | A script that creates the labels for associated files |
| [Resample Point Cloud](https://github.com/Brent-Murray/point-dl/blob/main/utils/resample_point_clouds.py) | A script that resamples point clouds writing them out |
| [Tools](https://github.com/Brent-Murray/point-dl/blob/main/utils/tools.py) | A script with useful tools for point cloud deep learning and classification |
| [Train](https://github.com/Brent-Murray/point-dl/blob/main/utils/train.py) | A script that defines the training/validatoin/testing process |

### Examples
| Example | Description |
| ------- | ----------- |
| [DGCNN Main](https://github.com/Brent-Murray/point-dl/blob/main/dgcnn_main.py) | An example script to run a classification using the Dynamic Graph CNN model |
| [Dual Model Main](https://github.com/Brent-Murray/point-dl/blob/main/dual_model_main.py) | An example script to run a classificatoin using a dual model |
| [PointCNN Main](https://github.com/Brent-Murray/point-dl/blob/main/pointcnn_main.py) | An example script to run a classification using the PointCNN model |
| [PointNet++ Main](https://github.com/Brent-Murray/point-dl/blob/main/pointnet2_main.py) | An example script to run a classification using the PointNet++ model |
