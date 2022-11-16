# point-dl
Point based deep learning modules and associated tools for classification. There are [Pytorch Geometric](https://github.com/Brent-Murray/point-dl/tree/main/PyG) based models/tools as well as [Pytorch](https://github.com/Brent-Murray/point-dl/tree/main/Pytorch) based ones.

Contents
----
### Pytorch Geometric
### Models
| Model | Description | Reference |
| ----- | ----------- | --------- |
| [Classifier](https://github.com/Brent-Murray/point-dl/blob/main/PyG/models/classifier.py) | A script used for classification using outputs of other models | NA |
| [DGCNN](https://github.com/Brent-Murray/point-dl/blob/main/PyG/models/dgcnn.py) | Dynamic Graph CNN model architecture | [(Wang et al., 2019)](https://arxiv.org/abs/1801.07829) |
| [Dual Model](https://github.com/Brent-Murray/point-dl/blob/main/PyG/models/dual_model.py) | A script used to combine the outputs of two models based on defined method | NA |
| [PointCNN](https://github.com/Brent-Murray/point-dl/blob/main/PyG/models/pointcnn.py) | PointCNN model architecture | [(Hell et al., 2022)](https://link.springer.com/article/10.1007/s41064-022-00200-4) |
| [PointNet++](https://github.com/Brent-Murray/point-dl/blob/main/PyG/models/pointnet2.py) | PointNet++ model architecture | [(Qi et al., 2017)](https://arxiv.org/abs/1706.02413) |

### Utils
| Util | Description |
| ---- | ----------- |
| [Augmentation](https://github.com/Brent-Murray/point-dl/blob/main/PyG/utils/augmentation.py) | A script that performes augmentations on point clouds |
| [Tools](https://github.com/Brent-Murray/point-dl/blob/main/PyG/utils/tools.py) | A script with useful tools for point cloud deep learning and classification |
| [Train](https://github.com/Brent-Murray/point-dl/blob/main/PyG/utils/train.py) | A script that defines the training/validation/testing process |
| [Train Comp](https://github.com/Brent-Murray/point-dl/blob/main/PyG/utils/train_comp.py) | A script that trains the models for composition |

### Main Scripts
| Example | Description |
| ------- | ----------- |
| [Comp Main](https://github.com/Brent-Murray/point-dl/blob/main/PyG/comp_main.py) | The main script that runs models for species composition |
| [Dual Model Main](https://github.com/Brent-Murray/point-dl/blob/main/PyG/dual_model_main.py) | The main script to run a dual model for species composition |
| [HP Optim](https://github.com/Brent-Murray/point-dl/blob/main/PyG/hp_optim.py) | Script that tunes hyperparameters using Optuna |

### Pytorch
### Models
| Model | Description | Reference |
| ----- | ----------- | --------- |
| [Point Augment](https://github.com/Brent-Murray/point-dl/tree/main/Pytorch/models/PointAugment) | Adaptation of the Point Augment model (see [below](https://github.com/Brent-Murray/point-dl/blob/main/README.md#point-augment) for contents)| [(Li et al., 2020)](https://arxiv.org/abs/2002.10876) |
| [DGCNN](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/dgcnn.py) | A Pytorch implementation of DGCNN | [(Wang et al., 2019)](https://arxiv.org/abs/1801.07829) |
| [DGCNN Extended](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/dgcnn_extended.py) | An extended version of the Pytorch DGCNN model| NA |

### Point Augment
| Folder | File | Description |
| ------ | ---- | ----------- |
| Augment | [Augmentor](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/augment/augmentor.py) | The augmentor (generator) model |
| Common | [Loss Utils](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/common/loss_utils.py) | The loss functions for the adapted model |
| Models | [DGCNN](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/models/dgcnn.py) | Pytorch implementation of DGCNN |
| Utils | [Augmentation](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/utils/augmentation.py) | A script that performes augmentations on point clouds |
| Utils | [Tools](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/utils/tools.py) | A script with useful tools for point cloud deep learning and classification |
| Utils | [Train](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/utils/train.py) | A script that defines the training/validation/testing process |
| Point Augment | [Main](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/main.py) | The main scipt that runs the model |

### Utils
| Util | Description |
| ---- | ----------- |
| [Augmentation](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/utils/augmentation.py) | A script that performes augmentations on point clouds |
| [Tools](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/utils/tools.py) | A script with useful tools for point cloud deep learning and classification |
| [Train](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/utils/train.py) | A script that defines the training/validation/testing process |

### Main Scripts
| Example | Description |
| ------- | ----------- |
| [HP Tuner](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/hp_tuner.py) | Script that tunes hyperparameters using Optuna |
| [Main](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/main.py) | The main script that runs models |
