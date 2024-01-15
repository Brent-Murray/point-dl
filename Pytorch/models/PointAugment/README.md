# PointAugment 
This is the implementation of the preprocessing and modeling of [Estimating tree species composition from airborne laser scanning data using point-based deep learning models](https://www.sciencedirect.com/science/article/pii/S0924271623003453?via%3Dihub).

This paper can be cited as:

Murray, B. A., Coops, N. C., Winiwarter, L., White, J. C., Dick, A., Barbeito, I., & Ragab, A. (2024). Estimating tree species composition from airborne laser scanning data using point-based deep learning models. ISPRS Journal of Photogrammetry and Remote Sensing, 207, 282–297. https://doi.org/10.1016/j.isprsjprs.2023.12.008



Contents
----
| Folder | Script | Description |
| ------ | ------ | ----------- |
| [augment](https://github.com/Brent-Murray/point-dl/tree/main/Pytorch/models/PointAugment/augment) | [augmentor.py](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/augment/augmentor.py) | The augmentor (generator) model used |
| [checkpoints](https://github.com/Brent-Murray/point-dl/tree/main/Pytorch/models/PointAugment/checkpoints/pretrained) | [dgcnn_pa_weights.t7](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/checkpoints/pretrained/dgcnn_pa_weights.t7) | Pretrained model using DGCNN with PointAugment |
| [checkpoints](https://github.com/Brent-Murray/point-dl/tree/main/Pytorch/models/PointAugment/checkpoints/pretrained) | [dgcnn_weights.t7](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/checkpoints/pretrained/dgcnn_weights.t7) | Pretrained model using DGCNN without PointAugment |
| [checkpoints](https://github.com/Brent-Murray/point-dl/tree/main/Pytorch/models/PointAugment/checkpoints/pretrained) | [pn2_pa_weights.t7](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/checkpoints/pretrained/pn2_pa_weights.t7) | Pretrained model using PointNet++ with PointAugment |
| [checkpoints](https://github.com/Brent-Murray/point-dl/tree/main/Pytorch/models/PointAugment/checkpoints/pretrained) | [pn2_weights.t7](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/checkpoints/pretrained/pn2_weights.t7) | Pretrained model using PointNet++ without PointAugment |
| [common](https://github.com/Brent-Murray/point-dl/tree/main/Pytorch/models/PointAugment/common) | [loss_utils.py](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/common/loss_utils.py) | The loss functions for the adapted models |
| [models](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/models) | [dgcnn.py](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/models/dgcnn.py) | A Pytorch adaptation of the DGCNN model for tree species composition estimation |
| [models](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/models) | [pointnet2.py](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/models/pointnet2.py) | A Pytorch adaptation of the PointNet ++ model for tree species composition estimation |
| [utils](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/utils) | [augmentation.py](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/utils/augmentation.py) | Manual point cloud augmentations |
| [utils](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/utils) | [send_telegram.py](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/utils/send_telegram.py) | A script to send telegram messages |
| [utils](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/utils) | [tools.py](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/utils/tools.py) | Usefull tools for point cloud deep learning and classification |
| [utils](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/utils) | [train.py](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/utils/train.py) | Training, validation and testing script |
| [PointAugment](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment) | [main.py](https://github.com/Brent-Murray/point-dl/blob/main/Pytorch/models/PointAugment/main.py) | The main script to run the model and define parameters |
