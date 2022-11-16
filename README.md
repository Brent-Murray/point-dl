# point-dl
Point based deep learning modules and associated tools for classification. There are [Pytorch Geometric](https://github.com/Brent-Murray/point-dl/tree/main/PyG) based models/tools as well as [Pytorch](https://github.com/Brent-Murray/point-dl/tree/main/Pytorch) based ones.

Contents
----
### Pytorch Geometric
### Models
| Model | Description |
| ----- | ----------- |
| [Classifier](https://github.com/Brent-Murray/point-dl/blob/main/PyG/models/classifier.py) | A script used for classification using outputs of other models |
| [DGCNN](https://github.com/Brent-Murray/point-dl/blob/main/PyG/models/dgcnn.py) | Dynamic Graph CNN model architecture | 
| [Dual Model](https://github.com/Brent-Murray/point-dl/blob/main/PyG/models/dual_model.py) | A script used to combine the outputs of two models based on defined method |
| [PointCNN](https://github.com/Brent-Murray/point-dl/blob/main/PyG/models/pointcnn.py) | PointCNN model architecture |
| [PointNet++](https://github.com/Brent-Murray/point-dl/blob/main/PyG/models/pointnet2.py) | PointNet++ model architecture |

### Utils
| Util | Description |
| ---- | ----------- |
| [Augmentation](https://github.com/Brent-Murray/point-dl/blob/main/PyG/utils/augmentation.py) | A script that performes augmentations on point clouds |
| [Tools](https://github.com/Brent-Murray/point-dl/blob/main/PyG/utils/tools.py) | A script with useful tools for point cloud deep learning and classification |
| [Train](https://github.com/Brent-Murray/point-dl/blob/main/PyG/utils/train.py) | A script that defines the training/validation/testing process |
| [Train Comp](https://github.com/Brent-Murray/point-dl/blob/main/PyG/utils/train_comp.py) | A script that trains the models for composition |

### Examples
| Example | Description |
| ------- | ----------- |
| [Comp Main](https://github.com/Brent-Murray/point-dl/blob/main/PyG/comp_main.py) | The main script that runs models for species composition |
| [Dual Model Main](https://github.com/Brent-Murray/point-dl/blob/main/PyG/dual_model_main.py) | The main script to run a dual model for species composition |
| [HP Optim](https://github.com/Brent-Murray/point-dl/blob/main/PyG/hp_optim.py) | Script that tunes hyper parameters using Optuna |
