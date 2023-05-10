# UI-detection

### Dependencies

A requirement.txt will be ramped up in future updates.

Key dependencies include torch(2.0.0+cu117), torchvision(0.15.1), tqdm, and numpy.

### Dataset

[Enrico](https://github.com/luileito/enrico) dataset is intended to be used in this project.

To access the data, there are two potential ways of aquiring the data. You can attempt to download the data from the [enrico repository](https://github.com/luileito/enrico). Alternatively, if you are using Linux or WSL, you can access the dataset by using the **download_data.sh** in enrico_utils folder. The enrico_utils folder contains a very small subset of the [MultiBench](https://github.com/pliang279/MultiBench) toolkit, which stands for Multiscale Benchmarks for Multimodal Representation Learning. Check out its dedicated repo for more details.

### Method

The methods can be found in the folder model_zoo, where all the model structures are attempted there.

- VGG16

  - 20 epoch training with SGD (a preliminary attempt): top 1 accuracy is **25.90%** (random guess is 5%, since there are 20 classes for whole page view types)
  - This is still a temporary result. According to [Luis A. Leiva, Asutosh Hota and Antti Oulasvirta](https://userinterfaces.aalto.fi/enrico/), the top 1 accuracy for a revised VGG16 model can be as high as 75.8%.

- Further attempts include:

  - model distillation and performance optimization to enable usage on mobile devices

  - higher accuracy for useablity (as high as possible, including finding out-of-domain dataset to test the generalizability of the model)

  - newer methods like incorporating attention

  - object detection to label all the functionals on the screen
  
### Model

The pretrained model will be made public when available. The model will be made public through web servies outside github, since the file is too large that requires git-lfs.

Check the training process using tensorboard:

```
tensorboard --logdir=runs
```

