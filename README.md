# Superpixel Sampling Networks

Pure PyTorch implementation of Superpixel Sampling Networks that does not require CUDA code compilation.
Some minor modifications were also made.

Forked from: https://github.com/perrying/ssn-pytorch

paper: https://arxiv.org/abs/1807.10174  
original code: https://github.com/NVlabs/ssn_superpixels

# Requirements
- PyTorch >= 1.4 (This fork was tested only with 1.12)
- scikit-image
- matplotlib

# Usage
## inference
SSN_pix
```
python inference.py --image /path/to/image
```
SSN_deep
```
python inference.py --image /path/to/image --weight /path/to/pretrained_weight
```

## training
```
python train.py --root /path/to/BSDS500
```

# Results
SSN_pix  
<img src=https://github.com/vvarga90/ssn-pytorch/blob/master/SSN_pix_result.png>

SSN_deep  
<img src=https://github.com/vvarga90/ssn-pytorch/blob/master/SSN_deep_result.png>
