# DPSIM

# Dynamic Prototype-guided Structural Information Maintaining for Unsupervised Domain Adaptation
This repo is the implementation of Dynamic Prototype-guided Structural Information Maintaining for Unsupervised Domain Adaptation.

## 2. Usage
### 2.1 Prepare data
#### Image Classification: Office-31, Office-Home
#### Object Detection: Cityscapes, FoggyCityscapes


### 2.2 Dependencies

	Python: 3.8.10
	PyTorch: 1.7.1
	Pillow: 9.5.0
	Torchvision: 0.8.2
	CUDA: 11.0
	NumPy: 1.22.4
	PIL: 7.2.0

### 2.3 Train

- Train
```
CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset clipart --net res101 --cuda \
--epochs 12 --gamma 5.0 --warmup --context --contrasiveloss --pairsimloss \
--alpha1 1.0 --alpha2 1.0 --alpha3 1.0 \
--lamda1 1.0 --lamda2 1.0 --lamda3 0.01 --lamda4 0.005 --lamda5 0.005 \
--num_aux1 2 --num_aux2 4 --desp 'DPSIM_con005_pair005' --cluster_loss 'Wt_KCL' --bs 1 --imgc 10000 
```

## Acknowledgement
Our code is based on the project [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library) and [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).

