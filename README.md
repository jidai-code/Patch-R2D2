# Patch-R2D2
### Abstract 

- This is a modification to the original R2D2 feature detector and descriptor network
- Patch-R2D2 generate R2D2 descriptor for a given patch image

- Please consider cite the original paper if you find this code useful

  @article{revaud2019r2d2,
  title={R2d2: Repeatable and reliable detector and descriptor},
  author={Revaud, Jerome and Weinzaepfel, Philippe and De Souza, C{\'e}sar and Pion, Noe and Csurka, Gabriela and Cabon, Yohann and Humenberger, Martin},
  journal={arXiv preprint arXiv:1906.06195},
  year={2019}
  }

- The implementation is based on original [git](https://github.com/naver/r2d2)

### Requirements

- You need Python 3.6+ equipped with standard scientific packages and PyTorch1.1+. Typically, conda is one of the easiest way to get started:

```
conda install python tqdm pillow numpy matplotlib scipy
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

### Usage

- Please see `demo_patch.py` for generating descriptor for a given patch
- Please see `demo_verification.py` for performance comparison with the original R2D2 descriptor

