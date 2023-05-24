# UI2Vec: Towards Multimodal Embedding on UI classification and description generation

## What is the fuss?

Current UI classification systems often use computer vision only, and their accuracy is usually about 0.75, far from usable in real life scenarios. Texts on UI screens can often explicitly present the functionality of the page, therefore an ideal source of data that we should not miss. In this work, we attempt to utilize the two modalities: vision and text, for embedding-based UI classification.

Besides, the texts extracted from images can also be a great source to try to enable machines to dynamically describe the what abouts on the screen. If that has been made a reality, it could be potentially a great way of increasing accessibility among the elders and the impaired who might need the assistance to understand a UI page.

## Dependencies

Key dependencies include torch(2.0.0+cu117), torchvision(0.15.1), tqdm, and numpy.

## Dataset

[Enrico](https://github.com/luileito/enrico) dataset is intended to be used in this project. Enrico is a subset of a large UI dataset [Rico](https://interactionmining.org/rico), with selection and relabeling. We also plan to use Rico to train our method in the future, since the scale is much larger, thus potentially more representative.

To access the data, there are two potential ways of aquiring the data. You can attempt to download the data from the [enrico repository](https://github.com/luileito/enrico). Alternatively, if you are using Linux or WSL, you can access the dataset by using the **download_data.sh** in enrico_utils folder. The enrico_utils folder contains a very small subset of the [MultiBench](https://github.com/pliang279/MultiBench) toolkit, which stands for Multiscale Benchmarks for Multimodal Representation Learning. Check out its dedicated repo for more details.

All the 20 classes are: bare, dialer, camera, chat, editor, form, gallery, list, login, maps, mediaplayer, menu, modal, news, other, profile, search, settings, terms, and tutorial. (encoded as from 0-19)

## Models

### Pretrained CV models

- VGG16

  - 252 epoch training with Adam: top-k accuracy (1-3-5) is **38.356%**, 61.644% and 73.288% respectively. (random guess is 5%, since there are 20 classes for whole page view types)
  
  - According to [Luis A. Leiva, Asutosh Hota and Antti Oulasvirta](https://userinterfaces.aalto.fi/enrico/), the top 1 accuracy for a revised VGG16 model can be as high as 75.8%.
  
    top1-acc: 79.110%
    top3-acc: 95.890%
    top5-acc: 98.630%
- VGG-short
  - 118 epoch training with Adam: top-k accuracy (1-3-5) is **38.699%**, 60.616%, 72.603% respectively.
  - This VGG-short only has 6 convolutional layers, yet it performs just as good as VGG 16. However, this net is extensively larger in parameter, since the flattened vector is extremely long.

- ViT
  - 55 epoch training with Adam: top 1 accuracy is around **21%**
  - It seems that ViT is prone to underfit, probably because of the large model size and not a lot to "focus" on (attention), we also experienced gradient explosion during training.

- Further attempts include:
  - model distillation and performance optimization to enable usage on mobile devices
  
  - higher accuracy for useablity (as high as possible, including finding out-of-domain dataset to test the generalizability of the model)
  
  - newer methods like incorporating attention
  
  - object detection to label all the functionals on the screen
  
### OCR for text extraction

There are numerours models of this kind. For keeping it simple, we do not attempt to retrain one, instead, we intend to use [easyOCR](https://github.com/JaidedAI/EasyOCR) as our model for extracting words from images.

Granted, the words obviously needed some semantic grouping, but to keep it simple, we should probably do it in the future.

### Embedding (or modal fusion)

Train it on potentially two scopes:

- simple classification, with classes used as supervision (cross-entropy loss)
- dimensional embedding, first augment the dataset by creating <base, similar, far> tuples, then try to minimize base and similar (belonging to the same group of class) and maximize base, far (belonging to different class). If we have N training samples, each class select M as similar, and a random other class UI (19), then the training set size would be N * M * 19, very large in scale indeed.

### NLG (Natual Language Generation)

Natual Language Generation is all the hype these days. We attempt to use the semantics from the OCR and the class of each UI to generate a valid introduction to the UI page. The method is still under investigation, yet I want to keep it very simple. Even a template matching generation could work.

Besides all that, the intro can also be in the form of multimedia, with video tutorials in place.

### Misc

Check the training process using tensorboard:

```
tensorboard --logdir=runs
```

