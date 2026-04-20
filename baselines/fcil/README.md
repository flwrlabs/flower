# Official Pytorch Implementation for GLFC

## [[CVPR-2022] Federated Class-Incremental Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Dong_Federated_Class-Incremental_Learning_CVPR_2022_paper.html)

This is the implementation code of the CVPR 2022 paper ["**Federated Class-Incremental Learning**"](https://cvpr2022.thecvf.com/). 

You can also find the arXiv version with supplementary materials [here](https://arxiv.org/abs/2203.11473). More related works are provided at [Dynamic Federated Learning](https://github.com/conditionWang/Dynamic-FL), please work with us to make FL more practical and realistic.


## Framework:

![overview](./fig/overview.png)


## Prerequisites:

    * python == 3.6
    * torch == 1.2.0
    * numpy
    * PIL
    * torchvision == 0.4.0
    * cv2
    * scipy == 1.5.2
    * sklearn == 0.24.1

## Datasets:

* **CIFAR100:** You don't need to do anything before running the experiments on [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

* **Imagenet-Subset (Mini-Imagenet):** Please manually download the on [Imagenet-Subset (Mini-Imagenet)](https://github.com/yaoyao-liu/mini-imagenet-tools) dataset from the official websites, and place it in './train'.

* **Tiny-Imagenet:** Please manually download the on [Tiny-Imagenet](https://github.com/seshuad/IMagenet) dataset from the official websites, and place it in './tiny-imagenet-200'.



## Training:

* Please check the detailed arguments in './src/option.py'.

```shell
python fl_main.py
```


## Performance:

* **Experiments on CIFAR100 dataset**

![cifar](./fig/imagenet_subset_result.png)

* **Experiments on Imagenet-Subset (Mini-Imagenet) dataset**

![imagenet-subset](./fig/cifar_result.png)

## Related Works

We apply federated class-incremental learning to semantic segmentation task.

1. [[CVPR-2023] Federated Incremental Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2023/html/Dong_Federated_Incremental_Semantic_Segmentation_CVPR_2023_paper.html) [[Code]](https://github.com/JiahuaDong/FISS)

## Citation:

If you find this code is useful to your research, please consider to cite our paper.

```
@InProceedings{dong2022federated,
    author = {Dong, Jiahua and Wang, Lixu and Fang, Zhen and Sun, Gan and Xu, Shichao and Wang, Xiao and Zhu, Qi},
    title = {Federated Class-Incremental Learning},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2022},
}
```

## Contact:

* **Lixu Wang:**  lixuwang2025@u.northwestern.edu
* **Jiahua Dong:** dongjiahua@sia.cn


