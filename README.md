# Learning Motion Representation for Real-Time Spatio-Temporal Action Localization
An Pytorch implementation of our work.

We built our work based on Pytorch implementation of [Online Real-time Multiple Spatiotemporal Action Localisation and Prediction](https://arxiv.org/pdf/1611.08563.pdf).

## Environment

- Ubuntu 16.04
- Python 3.6
- CUDA 8.0
- CuDNN 7.1
- Pytorch 0.4.0
- Opencv 3.4
- Matlab 2016b (if you need to compute the video-frame level)

## Training

We use the official Pytorch implementation of [PWC-Net](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch) as our flow subnet. (Notes: The PWC-Net repo is developed using Python 2.7 & Pytorch 0.2.0 & CUDA 8.0. We test several configurations to use the Pytorch implementation. Current environment can run this code correctly). You can use train-*.py scripts to train the whole network (we recommend to use `train-ucf24-apex.py` which is much faster but a little accuracy drop).

We use 4 GTX 1080ti graphics cards to train the network with 32 batch-sizes.

## Testing

### Frame-level 
You can use `val-ucf24.py` to evaluate the frame-level mAP

### Video-level
The video-level evaluation coda is in `./matlab-online-display`. You can run `myI01onlineTubes.m` to produce the video-level results.

## References
- [1] Wei Liu, et al. SSD: Single Shot MultiBox Detector. [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [2] S. Saha, G. Singh, M. Sapienza, P. H. S. Torr, and F. Cuzzolin, Deep learning for detecting multiple space-time action tubes in videos. BMVC 2016 
- [3] G. Singh, S Saha, M. Sapienza, P. H. S. Torr and F Cuzzolin. Online Real time Multiple Spatiotemporal Action Localisation and Prediction. ICCV, 2017.
- [4] Deqing Sun and Xiaodong Yang and Ming-Yu Liu and Jan Kautz. PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume. CVPR, 2018.
- [5] Liu, Songtao and Huang, Di and Wang, and Yunhong. Receptive Field Block Net for Accurate and Fast Object Detection. The European Conference on Computer Vision (ECCV).
- [Original SSD Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A huge thanks to Max deGroot, Ellis Brown for Pytorch implementation of [SSD](https://github.com/amdegroot/ssd.pytorch).
- A huge thanks to Gurkirt Singh for Online Real-time Multiple Spatiotemporal Action Localisation and Prediction Pytorch implementation [ROAD](https://github.com/gurkirt/realtime-action-detection).
 
