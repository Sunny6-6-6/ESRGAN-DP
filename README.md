# ESRGAN-DP: Enhanced Super-Resolution Generative Adversarial Network with Adaptive Dual Perceptual Loss
Jie Song, Huawei Yi, Wenqian Xu, Xiaohui Li, Bo Li, Yuanyuan Liu
> Reference Toolbox: Xintao Wang, Ke Yu, Kelvin C.K. Chan, Chao Dong and Chen Change Loy. BasicSR. https://github.com/xinntao/BasicSR, 2020.


1. Put trainset and testset into /datasets.

2. Put resnet50.pth and vgg19.pth into /basicsr/model/pretraining network.

3. Train:

    ```
    python basicsr/train.py -opt options/train/ESRGAN_DP/train_ESRGAN_DP.yml
   
    ```
4. Test:

    ```
    python basicsr/test.py -opt options/test/ESRGAN_DP/test_ESRGAN_DP.yml
   
    ```
