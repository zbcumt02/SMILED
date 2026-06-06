# SMILED
A Subgraph-based Multi-label Learning Method with Instance-Label Ensemble for Denoising

![model](./method.png)

If using this code, please cite our paper.

```
@article{
}
```

## System requirement

#### Programming language
Python 3.5 +

#### Python Packages
Pytorch 2.0.0+cu118, Numpy 1.24.4, Networkx 3.1, PyTorch Geometric 1.7.2, Scikit-learn 1.3.2

#### Datasets

Raw datasets are obtained from https://mulan.sourceforge.net/datasets-mlc.html. Datasets are processed and saved into mat file.

## Training 

#### Train the network

```
cd Ours/Python
python Main.py --data-name=yeast --num-epochs=100
```


## Acknowlegdements

Part of code borrow from https://github.com/muhanzhang/SEAL, https://github.com/muhanzhang/DGCNN and [Line Graph Neural Networks for Link Prediction](https://arxiv.org/pdf/2010.10046.pdf). Thanks for their excellent work!
