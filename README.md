# UI2Vec: Towards Multimodal Embedding on UI classification and description generation

## What is all the fuss?

Current UI classification systems often use computer vision only, and their accuracy is usually about 0.75, far from usable in real life scenarios. Texts on UI screens can often explicitly present the functionality of the page, therefore an ideal source of data that we should not miss. In this work, we attempt to utilize the two modalities: vision and text, for embedding-based UI classification.

Besides, the texts extracted from images can also be a great source to try to enable machines to dynamically describe the what abouts on the screen. If that has been made a reality, it could be potentially a great way of increasing accessibility among the elders and the impaired who might need the assistance to understand a UI page.

## Folders

- **enrico_utils**: everything that is related to the manipulation of the enrico dataset, including embedding dataset generation, dataloader and source data downloader
- **inference**: inference of the models, including GloVe, VGG16 and UI2Vec
- **models**: every model implementation, in PyTorch
- **test**: for testing the performance of UI2Vec model

The training and evaluation of vision models and UI2Vec is at the root folder files.

## Dependencies

Key dependencies include python(3.8), torch(2.0.0+cu117/cu118), torchvision(0.15.1), gensim etc.

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
| ViT                  | 327                 | 20.5          | 44.5          | 62.0          |
| VGG16 (Enrico paper) | -                   | 75.8          | 88.4          | 92.9          |

Note that in this split, we only trained on the train set, validation set is not used as training data. Thus, due to the imbalance of between classes, the results are pretty much split-dependent. Because of that, the author's of Enrico must have experimented several splits to get a "good split" which evenly split all the classes in all three sets.

To replicate the result in the paper, and to provide an embedding model, we used validation split from our previous split as supervision, and **all data** as training set (not a good way if you are using the CV model only). This is prone to overfitting the data and have weak generalization capability. In this training version, we achieved an accuracy (top-1, 3, 5 respectively) of **79.110%**, 95.890% and 98.630%. According to the loss and validation accuracy, the model have converged since that.

However, since we are only using the embedding from this CV model,there is still opportunity to correct the potential overfitting after going through unsupervised loss (mention below) for embedding fusion and downsampling. We extract the penultimate layer (an fc-layer) before the output layer as our embedding output. The dimension is set to **768**.

### OCR for text extraction

There are numerours models of this kind. The information we need from this process is the content on each screen and the position of the text. [easyOCR](https://github.com/JaidedAI/EasyOCR) seems fit for this task. Granted, the words obviously needed some semantic grouping, but to keep it simple, we should probably do it in the future.

We record the top-left position of the text and the content of the text in csv files. The results will be made available when possible. To get a unifided position, we first reshape all the UI pictures to 800 * 400 (heigth * width). Besides, to avoid grasping information that is not context-dependent (eg. time, battery life etc.), we stripped the OCR result with height below 20px.

After that, we need to generate the embedding for the texts. However, this is difficult since there are errors (quite a few) in the recognition, so pretrained models such as word2vec would fail to generate the embedding.

For the record, the pre-trained model we use is from [gensim-data](https://github.com/RaRe-Technologies/gensim-data). To be specific, the **glove-wiki-gigaword-100** model. This might not be the best choice, yet 400000 distict words is good-enough to validate the idea.

To address the no embedding issue, we ran a word-by-word sanity check which replaces the errors (mostly spelling error) with a word from the glove dictionary. After that, for each UI sample, we generated an embedding for that sample with dimension **100**. In case of multiple words, we used average embedding method, and for no-word cases, we simply leave the embedding as a zero vector.

### Embedding (or modal fusion)

After we aquired the CV and NLP embedding, the next step to do is to train the actual embedding for a certain UI.

Concatenation is the simplest way of fusion. We did the concatenation based on normalized embeddings, resulting in a 868-dimensional vector.

After concatenation, we made a dataset augmentation, to create the training data for our encoder. The training is based on triplet-loss, thus we need to find an anchor vector, a positive vector and a negative vector. Here, to balance out the class count differences, we expanded every class to 1000 entries, making it a total of 20,000 training samples. The augmentation algorithm is as follow:

```pseudocode
for ui_class in classes:
	for each anchor in ui_class:
		randomly choose another item (positive) as the positive vector
		randomly choose another ui_class, from other_class, randomly choose an item as negative
		build sample tuple <anchor, positive, negative>
		if all the items have drained in the ui_class, yet the total sample has not reach 1000:
			start again from the first anchor
		else:
			continue to another class
```

Train the new dataset until converge with a 3-layer (including input and output) MLP. The output embedding will have **150** dimensions.

### Downstream Tasks

The downstream task include classification of UIs, retrieval of similar UIs and more.

In this example, we use an SVM model to classify the UIs. We splitted 20% of the samples as test set, and used scikit-learn's vanilla SVC(probability=True) to train the model.

We are happy to report that accuracy can be as high as 95.55%, and top-3 accuracy is 100%. For this accuracy, it is totaly useable in real-life applications.

Note that this result is only representative on Enrico dataset, a relatively small dataset, thus the model may not be generalizable. However, it is testimony that the embedding method does work, since it boosted the classification accuracy greatly.

Further work include abalation study on whether either nlp embedding or cv embedding can be deleted, or if the encoder part is actually useful.

### NLG (Natual Language Generation)

Natual Language Generation is all the hype these days. We attempt to use the semantics from the OCR and the class of each UI to generate a valid introduction to the UI page. The method is still under investigation, yet I want to keep it very simple. Even a template matching generation could work.

Besides all that, the intro can also be in the form of multimedia, with video tutorials in place.

### Misc

Check the training process using tensorboard:

```
tensorboard --logdir=runs
```

