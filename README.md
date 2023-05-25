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

The feature space for UIs are much smaller compared to natural images. In experiment (3 experiments), we found out that VGG 16 (not exactly the original, but almost the same) perform just as well as shallower VGG (by cutting the convolution layers in half) and much better than ViT.

Here are the results (trained with the same split, with train:val:test = 65:15:20, as [Luis A. Leiva, Asutosh Hota and Antti Oulasvirta](https://userinterfaces.aalto.fi/enrico/)) did in their paper:

|                      | Parameter Size (MB) | Top-1 Acc (%) | Top-3 Acc (%) | Top-5 Acc (%) |
| -------------------- | ------------------- | ------------- | ------------- | ------------- |
| VGG16                | **154**             | 38.356        | **61.644**    | **73.288**    |
| VGG16-shallow        | 403                 | **38.699**    | 60.616        | 72.603        |
| ViT                  | 327                 | ~21           | -             | -             |
| VGG16 (Enrico paper) | -                   | 75.8          | 88.4          | 92.9          |

Note that in this split, we only trained on the train set, validation set is not used as training data. Thus, due to the imbalance of between classes, the results are pretty much split-dependent. Because of that, the author's of Enrico must have experimented several splits to get a "good split" which evenly split all the classes in all three sets.

To replicate the result in the paper, and to provide an embedding model, we used validation split from our previous split as supervision, and **all data** as training set (not a good way if you are using the CV model only). This is prone to overfitting the data and have weak generalization capability. In this training version, we achieved an accuracy (top-1, 3, 5 respectively) of **79.110%**, 95.890% and 98.630%. According to the loss and validation accuracy, the model have converged since that.

However, since we are only using the embedding from this CV model,there is still opportunity to correct the potential overfitting after going through unsupervised loss (mention below) for embedding fusion and downsampling. We extract the penultimate layer (an fc-layer) before the output layer as our embedding output. The dimension is set to **768**.

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

