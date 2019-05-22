# [Checkerboards](<https://arxiv.org/pdf/1501.05759.pdf>) summary for weekly

Modern classical pedestrian detection methods utilize handcrafted image representation such as HOG features[^1] and Haar filters[^2], this method has proved as inferior in recent years to the learned representation offered by modern deep learning methods [^3]. These learned representation have the benefit of allowing for a more global understanding of the image space and a removal of the need of a costly sliding window mechanism scanning all levels. 

This simplification may come with some computational burden of it's own which might be surpassed using a hybrid mode which combines the simplicity of hand crafted features and window proposal aided by a modern module. A current effort is taken in this hybrid direction by the ACV team, in order to get some insight into our performance, we  compare our method to top-performer among the classical PD methods named Checkerboards [^4] , in order to make this comparison complete, we make a three fold comparison of our method, checkerboard method and and combined one with our proposals replacing only the sliding window mechanism. 

[^1]: Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." *CVPR (1)* 1 (2001): 511-518.
[^2 ]: Dalal, Navneet, and Bill Triggs. "Histograms of oriented gradients for human detection." *international Conference on computer vision & Pattern Recognition (CVPR'05)*. Vol. 1. IEEE Computer Society, 2005.

[^3]: Liu, Wei, et al. "Learning Efficient Single-stage Pedestrian Detectors by Asymptotic Localization Fitting." *Proceedings of the European Conference on Computer Vision (ECCV)*. 2018.

[^4]: Zhang, Shanshan, Rodrigo Benenson, and Bernt Schiele. "Filtered channel features for pedestrian detection." *CVPR*. Vol. 1. No. 2. 2015.

## Method

### Summary of the Checkerboards method

Checkerboards method unifies a few top performing methods like Integral channel features [^5] and Informed haar-like features [^6],  it uses a representation of HOG+LUV features and a a fixed size sliding window mechanism across multiple pyramid levels, each such window is filtered by all 39 haar filters in order to create a single window feature vector. Each window is classified as pedestrian or background by a boosted forest followed by an NMS stage.

Parameters values vary between different tasks (such as image size, common pedestrian size etc.), in our implementation parameters were chosen in order to correspond to our hybrid implementation. A few quantities worth staging are the number of windows classified (~700,000), size of each feature vector (~60,000), minimal window size ([68,34]), number of scales (21) and stride factor in each pyramid level (6).

[^5]: P. Dollár, Z. Tu, P. Perona, and S. Belongie. Integralc hannel features. In BMVC, 2009
[^6]: S. Zhang, C. Bauckhage, and A. B. Cremers. Informed haar-like features improve pedestrian detection. In CVPR, 2014

## Results, Discussion and Analysis

### SIRC v0.29 Vs Checkerboards Vs Checkerboards-SIRC v0.29 hybrid  (untrained to Citypersons)

Evaluation of the above three methods is presented below, # TODO - discuss. 

![](Y:\Code\Checkerboards\Checkerboards_CVPR15_codebase\figures\Figure_v0.29.2.png)



