#!/bin/bash

set -x
set -e

CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset clipart --net res101 --cuda \
--epochs 12 --gamma 5.0 --warmup --context --contrasiveloss --pairsimloss \
--alpha1 1.0 --alpha2 1.0 --alpha3 1.0 \
--lamda1 1.0 --lamda2 1.0 --lamda3 0.01 --lamda4 0.005 --lamda5 0.005 \
--num_aux1 2 --num_aux2 4 --desp 'DPSIM_con005_pair005' --cluster_loss 'Wt_KCL' --bs 1 --imgc 10000 
