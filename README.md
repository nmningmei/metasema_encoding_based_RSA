# Encoding based RSA

# Models:
## Computer vision models
| VGG19 | Mobilenet-v2 | Resnet50 |
| :---: | :---: | :---: |
| <img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/cv_features/vgg19.jpg" width="300" height="300"> | <img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/cv_features/mobilenet.jpg" width="300" height="300"> | <img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/cv_features/resnet50.jpg" width="300" height="300"> |
## Word embedding models
| Fast Text | GloVe | Word2vec-2012 |
| :---: | :---: | :---: |
| <img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/word2vec_features/fasttext(light).jpg" width="300" height="300"> | <img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/word2vec_features/glove(light).jpg" width="300" height="300"> | <img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/word2vec_features/word2vec(light).jpg" width="300" height="300"> |

# Results
---
## Group-level average - 10mm RSA - baseline-1
| Correlation | Randomise p values|
|:---: | :---:|
|<img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/RSA_basedline_average_10mm_standard_group_average/group%20average.jpg" width="400"   height="400" title="RSA grouped average, N = 27">|<img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/RSA_basedline_average_10mm_standard_randomise/group%20average%20p%20values.jpg" width="400"   height="400" title="Permutation one-sample group test, TFCE corrected">|

## Group-level average - 10mm encoding - baseline-2
| Correlation | Randomise p values|
|:---: | :---:|
|<img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/encoding_group_average/group%20average.jpg" width="400"   height="400" title="RSA grouped average, N = 27">|<img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/encoding_randomise/group%20average%20p%20values.jpg" width="400"   height="400" title="Permutation one-sample group test, TFCE corrected">|

## Group-level average - 10mm encoding-based RSA
| R2 score | Randomise p values|
|:---: | :---:|
|<img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/encoding_based_RSA_10mm_group_average/group%20average.jpg" width="400"   height="400" title="RSA grouped average, N = 27">|<img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/encoding_based_RSA_10mm_randomise/group%20average%20p%20values.jpg" width="400"   height="400" title="Permutation one-sample group test, TFCE corrected">|

## Group-level average - 10mm encoding-based + feature selection RSA
| Correlation | Randomise p values|
|:---: | :---:|
|<img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/encoding_based_FS_RSA_10mm_group_average/group%20average.jpg" width="400"   height="400" title="RSA grouped average, N = 27">|<img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/encoding_based_FS_RSA_10mm_randomise/group%20average%20p%20values.jpg" width="400"   height="400" title="Permutation one-sample group test, TFCE corrected">|

## Group-level average - 10mm decoding-encoding-based RSA
| Correlation | Randomise p values|
|:---: | :---:|
|<img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/decoding_based_RSA_10mm_group_average/group%20average.jpg" width="400"   height="400" title="RSA grouped average, N = 27">|<img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/decoding_based_RSA_10mm_randomise/group%20average%20p%20values.jpg" width="400"   height="400" title="Permutation one-sample group test, TFCE corrected">|

## Group-level average - 10mm encoding-based brain-to-brain RSA
| Correlation | Randomise p values|
|:---: | :---:|
|<img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/encoding_based_2brains_RSA_10mm_group_average/group%20average.jpg" width="400"   height="400" title="RSA grouped average, N = 27">|<img src="https://github.com/nmningmei/metasema_encoding_based_RSA/blob/main/figures/encoding_based_2brains_RSA_10mm_randomise/group%20average%20p%20values.jpg" width="400"   height="400" title="Permutation one-sample group test, TFCE corrected">|
