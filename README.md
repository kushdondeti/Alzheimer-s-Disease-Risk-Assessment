# Alzheimer-s-Disease-Risk-Assessment_Using_AI/ML
Risk assessment for developing Alzheimer's Disease based on CT scans of brains. Determines the level of dementia: non-demented, mild-demented, very-mild-demented, and moderate-demented

Abstract:

Early detection is crucial for effective management of Alzheimer’s disease (AD) and screening for mild cognitive impairment (MCI) is common practice. Among several deep-learning techniques that have been applied to assessing structural brain changes on magnetic resonance imaging (MRI), convolutional neural network (CNN) has gained popularity due to its superb efficiency in automated feature learning with the use of a variety of multilayer perceptrons. We used a subset of a publicly available dataset of Brain MRI images with Mild Demented, Moderate Demented, Non-Demented, Very Mild Demented cases. Then, we used a technique called deep learning to build a classifier to distinguish between Non-Demented and others from MRI images. This technique focuses on deep artificial neural networks which are inspired by the human brain and can learn from data to perform tasks. Nowadays, these models can perform so efficiently in solving real-world problems. Since medical data sets have limited numbers of samples and training deep learning models with millions of parameters is so computationally expensive, training a deep neural network classifier from scratch is cumbersome. So, we chose a pre-trained deep neural network and optimized it for our task with a few steps of training on MRI images. In our results, we showed that the model can detect Alzheimer's from brain MRI images with high accuracy.


Project Summary:

One efficient and early way of detecting Alzheimer’s is to use Artificial Intelligence from brain MRI images. In this project, we have developed a machine learning model to perform this task. The problem is a classification problem with three classes where we distinguish between f different cases. There are several advantages of using Artificial Intelligence (AI) to tackle this problem:
It can be used for the early detection of Alzheimer’s.
It is less expensive than traditional methods
It can be deployed easily anywhere.

To develop the Artificial Intelligence (AI) model, we benefit from deep learning algorithms to train a model on chest X-ray images. After the training, the model is able to detect COVID-19 cases with high accuracy.


Introduction:

Alzheimer's disease (AD), the most typical form of dementia, is a significant challenge for healthcare in the twenty-first century. An estimated 5.5 million people aged 65 and older are living with AD, and AD is the sixth-leading cause of death in the United States. The global cost of handling AD, including medical, social insurance, and wage loss to the subjects' families, was $277 billion in 2018 in the US, massively impacting the overall economy and stressing the U.S. health care system.

AD is an irreversible, growing brain disorder marked by a decline in cognitive functioning with no validated disease-modifying therapy. Thus, a vast deal of effort has been made to develop strategies for early detection, especially at pre-symptomatic stages in order to slow or prevent disease progression.  In particular, advanced neuroimaging techniques, such as magnetic resonance imaging (MRI), have been developed and used to identify AD-related structural and molecular biomarkers.

Accelerated advancement in neuroimaging methods has made it challenging to combine large-scale, high-dimensional multimodal neuroimaging data. Therefore, interest has increased quickly in computer-aided machine learning methods for integrative analysis.

To apply machine learning techniques, proper architectural design or pre-processing steps must be predefined. Classification studies using machine learning generally require four steps: feature extraction, feature selection, dimensionality reduction, and feature-based classification algorithm selection. These procedures need specific knowledge and various optimization stages, which may be time-consuming. Also, the reproducibility of these methods has been a problem.

To overcome these difficulties, deep learning, an emerging field of machine learning research that uses raw data to generate features through learning, is drawing significant attention in the field of large-scale, high-dimensional medical imaging analysis. Deep learning methods, such as convolutional neural networks (CNN), have been shown to outperform existing machine learning methods.



Method:

Dataset:

The data we used for this project was Brain MRI images. This data consists of Mild Demented, Moderate Demented, Non-Demented, Very Mild Demented cases. This dataset is hand-collected from various websites with each and every labels verified and is publicly available on the Kaggle. Kaggle is a website for data scientists and machine learning practitioners which allows users to find and publish datasets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Since the classes are imbalanced, we decided to perform over-sampling using SMOTE. SMOTE, Synthetic Minority Oversampling TEchnique, is one of the most widely used approaches to synthesize new examples. SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space, and drawing a new sample at a point along that line. Specifically, a random example from the minority class is first chosen. Then k of the nearest neighbors for that example are found (typically k=5). A randomly selected neighbor is chosen and a synthetic example is created at a randomly selected point between the two examples in feature space.

Finally, we preprocess the data by scaling the between 0 and 1. To achieve more robustness and to avoid overfitting to the training data, we applied several types of data augmentations using rotation, zooming, flipping, and changing the brightness. Data augmentation in data analysis are a technique used to increase the amount of data by adding slightly modified copies of already existing data or newly created synthetic data from existing data. It acts as a regularizer and helps reduce overfitting when training a machine learning model.

Model Architecture:

As the next step, we created a deep learning model that is going to learn the difference between Mild Demented, Moderate Demented, Non-Demented, Very Mild Demented from MRI images. Deep learning is a class of machine learning algorithms that are a model of the human brain and uses multiple layers to progressively extract higher-level features from the raw input. For example, in image processing, lower layers may identify edges, while higher layers may identify the concepts relevant to a human such as digits or letters, or faces.

To train deep learning models, we need thousands or millions of samples in our dataset. Besides, a lot of computational resources are needed. So, since we didn’t have that much computational power and our medical dataset had much fewer samples, we used a technique named “Transfer Learning”. In this technique, we chose a pre-trained convolutional deep neural network to build our final model. 

Convolutional Neural Networks are one the best models to deal with image data. This pre-trained model has been trained on a very large data set on a general task for several days with high-performance computational resources. After the training, they have gained great knowledge and can extract meaningful patterns from the data. 

We chose VGG16 architecture as the base model to extract sufficient features from the data and only added a few layers to optimize the network for our task. So, in this setup, we extracted the patterns from chest X-ray images with the VGG16 model and only trained those added layers. Here is a summary of the architecture:

_________________________________________________________________
Layer (type)                 Output Shape              Param #
==============================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
_________________________________________________________________
average_pooling2d (AveragePooling2D) (None, 2, 2, 512)         0
_________________________________________________________________
flatten (Flatten)            (None, 2048)                     0
_________________________________________________________________
dense (Dense)                (None, 512)               1049088
_________________________________________________________________
dense_1 (Dense)              (None, 512)               262656
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 2052
==============================================================
Total params: 16,028,484
Trainable params: 1,313,796
_________________________________________________________________

The input to cov1 layer is of fixed size 224 x 224 RGB image. The image is passed through a stack of convolutional (conv.) layers, where the filters were used with a very small receptive field: 3×3 (which is the smallest size to capture the notion of left/right, up/down, center). In one of the configurations, it also utilizes 1×1 convolution filters, which can be seen as a linear transformation of the input channels (followed by non-linearity). The convolution stride is fixed to 1 pixel; the spatial padding of conv. layer input is such that the spatial resolution is preserved after convolution, i.e. the padding is 1-pixel for 3×3 conv. layers. Spatial pooling is carried out by five max-pooling layers, which follow some of the conv.  layers (not all the conv. layers are followed by max-pooling). Max-pooling is performed over a 2×2 pixel window, with stride 2.

Three Fully-Connected (FC) layers follow a stack of convolutional layers (which has a different depth in different architectures): the first two have 4096 channels each, the third performs 1000-way ILSVRC classification and thus contains 1000 channels (one for each class). The final layer is the soft-max layer. The configuration of the fully connected layers is the same in all networks.


Results:

After the training, we evaluated the performance of the model on 2560
samples that the model has never seen. In these samples, there were 662
Mild Demented, 624 Moderate Demented, 639 Non-Demented, 635
 Very Mild Demented cases.

In our evaluations, the model showed 92.71% accuracy, 98.29 AUC and 98% F1-score. Accuracy measures how many observations, both positive and negative, were correctly classified. AUC means the area under the curve of ROC. ROC is a chart that visualizes the tradeoff between true positive rate (TPR) and false positive rate (FPR). Basically, for every threshold, we calculate TPR and FPR and plot them on one chart. The higher TPR and the lower FPR is for each threshold the better and so classifiers that have curves that are more top-left-side are better. F1 score also combines precision and recall into one metric by calculating the harmonic mean between those two. The model could correctly classify 516 Moderate Demented samples, 524 Mild Demented samples, and 635 Very Mild Demented cases. Here is the result of the training and testing process:



References

Alzheimer's Association. "2018 Alzheimer's disease facts and figures." Alzheimer's & Dementia 14.3 (2018): 367-429.
Jo, Taeho, Kwangsik Nho, and Andrew J. Saykin. "Deep learning in Alzheimer's disease: diagnostic classification and prognostic prediction using neuroimaging data." Frontiers in aging neuroscience 11 (2019): 220.
De Strooper, Bart, and Eric Karran. "The cellular phase of Alzheimer’s disease." Cell 164.4 (2016): 603-615.
Galvin, James E. "Prevention of Alzheimer's disease: lessons learned and applied." Journal of the American Geriatrics Society 65.10 (2017): 2128-2133.
Schelke, Matthew W., et al. "Mechanisms of risk reduction in the clinical practice of Alzheimer’s disease prevention." Frontiers in aging neuroscience 10 (2018): 96.
Veitch, Dallas P., et al. "Understanding disease progression and improving Alzheimer's disease clinical trials: Recent highlights from the Alzheimer's Disease Neuroimaging Initiative." Alzheimer's & Dementia 15.1 (2019): 106-152.
Riedel, Brandalyn C., et al. "Uncovering biologically coherent peripheral signatures of health and risk for Alzheimer’s disease in the aging brain." Frontiers in aging neuroscience 10 (2018): 390.
Lu, Dengsheng, and Qihao Weng. "A survey of image classification methods and techniques for improving classification performance." International Journal of Remote sensing 28.5 (2007): 823-870.
Samper-González, Jorge, et al. "Reproducible evaluation of classification methods in Alzheimer's disease: Framework and application to MRI and PET data." NeuroImage 183 (2018): 504-521.
LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. "Deep learning." nature 521.7553 (2015): 436-444.
Plis, Sergey M., et al. "Deep learning for neuroimaging: a validation study." Frontiers in neuroscience 8 (2014): 229.
https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
https://en.wikipedia.org/wiki/Data_augmentation
https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images
https://neurohive.io/en/popular-networks/vgg16/
