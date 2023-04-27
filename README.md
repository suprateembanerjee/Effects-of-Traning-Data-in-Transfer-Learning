# Effects of Traning Data in Transfer Learning
#### Suprateem Banerjee | April 25, 2023

Based on the paper <b>A Data-Based Perspective on Transfer Learning</b> by <i>Jain et. al.</i>, which I presented earlier in the course, this project was about implementing the core concepts in this paper to verify the findings by the authors using a different dataset and different architectures from the ones mentioned in the paper.

## Introduction

The technique of transfer learning is used to adapt a pre-trained model on a source dataset to improve its performance on a downstream target task. This approach is used in various machine learning applications, but it is still unclear what drives the performance gains in transfer learning. Previous studies focused on the role of the source model, but there is evidence that the source dataset may also play an important role. The paper presents a framework for analyzing the impact of the source dataset's composition on transfer learning performance. The framework allows for the identification of subsets of the source dataset that positively or negatively affect downstream behavior and can be used to *remove* detrimental data from the source dataset to improve transfer learning performance. 

## Original Paper Setup

The paper uses a **Source Dataset** to train the model, and uses this model on a **Target Dataset** to analyze effects of removing specific classes (or groups of classes) on the performance on the target dataset.

**Source Dataset**: ImageNet (1.2M images across 10000 classes).

**Target Dataset**: CIFAR 10 (60000 images across 10 classes).

**Model**: ResNet-18 (Official PyTorch implementation).

## Techniques

**Fixed Feature Weights** This method freezes the weights for fine tuning the model for the target dataset using a different last layer, depending on the number of classes in target datasets.

**Subsampling** is used as opposed to **leave-one-out** as it better captures the similarity in patterns. While in **leave-one-out**, we exclude one class at a time, in **Subsampling**, we exclude a group of classes at a time. The authors explain this is better at removing related classes, such as various types of birds, if the patterns corresponding to birds is harming the prediction accuracy of a certain class in the target dataset.

## Methodology

### Computing Influences

The main idea in the paper revolves around computing influence that a subsampled set of classes have on the performance of the model on a target dataset. This has been achieved in four steps:

**Step 1**: Train a large number of model with subsets of classes removed.

**Step 2**: Fine tune those models on the target dataset (replacing the last layer to accomodate new class predictions).

**Step 3**: Estimate the influence of a source class on a target datapoint.

This influence in computed using a formula:

![image-20230426122324968](/Users/suprateembanerjee/Documents/1.png)

The algorithm specified can be captured in the following pseudocode:

```
def get_influences(
	S,  # source dataset
	k,  # number of classes in S
	T,  # target dataset
	n,  # number of classes in T
	m,  # number of models
	α):   # subset ratio

	S_subset = [shuffle(S) [:α * len (S)] for i in range (m)]

	for i in range (m):
		f[i] = train_model(A, S_subset [i])

	for k in range (K):
		for j in range (n):
			w = x = y = z = 0
			for i in range(m):
				if k in S_subset [i] ['label']:
					w += softmax(f[i] (T[j], S_subset[i]))
					× += 1 
				else:
					y += softmax(f[i] (T[j], S_subset [i]))
					z +=1
				influence[f'C[{k}] -> t[{j}]'] = w / × - y / z

	return influence
```

### Capabilities

The authors define three capabilities of their probing framework using the influence scores:

1. Identifying **granular target subpopulations** that correspond to the source classes
2. Debugging **Transfer Learning failures**
3. Detecting **Data Leakage** between the source and target datasets

**Capability 1**: The authors comment that even when the source class is not a direct sub-type of the target class, the downstream model can still leverage salient features from this class - such as shape or color - to predict on the target dataset.

![image-20230426202740779](/Users/suprateembanerjee/Documents/2.png)

![image-20230426203123605](/Users/suprateembanerjee/Documents/3.png)

**Capability 2**: The authors attempt to identify the most negatively influencing classes in the source dataset and remove it. They give the example of the ImageNet class Sorrel Horse and the CIFAR10 class Dog. The features from Sorrel Horse negatively impacts the prediction accuracy on the target dataset for the class Dog. Removing the Sorrel Horse class improves the prediction accuracy for Dog.

![image-20230426203545674](/Users/suprateembanerjee/Documents/4.png)

**Capability 3**: The authors comment that highly influential images across two datasets are often instances of data leakage or typically mislabeled. Influences here is a tool to determine any form of data leakage from the training set to the test set, which might interfere with understanding real performance of the model on the test set. 

The issue of misleading data is also considered. The authors use the most negatively influencing samples as an indicator to find training samples which are misleading the model. This , however, requires the technique to be used per datapoint in the training dataset.

![image-20230426203932464](/Users/suprateembanerjee/Documents/5.png)

![image-20230426204353393](/Users/suprateembanerjee/Documents/6.png)

## Personal Experimentation

### Set Up

**Training Dataset**: TinyImageNet

This dataset was released as a Kaggle Challenge, containing a mini version of the ImageNet dataset containing 1 Million 64x64 images across 200 classes (containing 500 images each).

**Target Dataset**: CIFAR-10

This dataset contains 60000 32x32 images across 10 classes (containing 6000 images each).

The model architecture used was **EfficientNet B3**.

### Observations and Results

My experiments showed similar observations with the paper. Similar classes across the two datasets showed strong influence.

For example,

Images from the CIFAR-10 class Dog showed **strong positive influence** from the following classes in **TinyImageNet**:

1. n02085620 Chihuahua
2. n02094433 Yorkshire terrier
3. n02099601 golden retriever
4. n02099712 Labrador retriever
5. n02106662 German shepherd, German shepherd dog, German police dog, alsatian
6. n02113799 standard poodle

However, the following classes in TinyImageNet had a **strong negative influence** on the class Dog in **CIFAR-10**:

1. n02437312 Arabian camel, dromedary, Camelus dromedarius
2. n02415577 bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis
3. n02129165 lion, king of beasts, Panthera leo
4. n01629819 European fire salamander, Salamandra salamandra

Removing the classes which had positive influence on the class Dog ended up reducing prediction accuracy of Dog from 0.124 to 0.057.

However, removing the classes which had strong negative influence on the class Dog ended up increasing prediction accuracy of Dog from 0.124 to 0.168.

This is in line with the results from the paper. 

## Discussions

While the paper is interesting and proves that the popular perception of larger datasets always training a better model may be false in transfer learning setups, there are several challenges that may arise. Some of these challenges are listed below:

1. In this method of analysis, it requires knowledge of the target dataset. In most real world applications, target dataset is unknown until deployed.
2. Removing some classes from training dataset, while being beneficial towards some classes, also harms performance on many others. Such compromises may not be ideal for larger object detection systems which need to detect a variety of classes.

Such challenges may prove to limit deployment of such systems in real world applications. However, it may still be relevant when we determine the probability of occurance of certain classes in certain scenarios. For instance, if the occurance of an Arctic Fox in an urban environment is deemed to be extremely unlikely, detection accuracies of Arctic fox, for instance, may be compromised to boost detection accuracies of a human, for instance.

## Conclusions

To conclude, this project implemented the concepts presented in the paper "A Data-Based Perspective on Transfer Learning" by Jain et al. using a different dataset and architecture. The paper proposed a framework to analyze the impact of the source dataset's composition on transfer learning performance and demonstrated that subsets of the source dataset can positively or negatively affect the downstream behavior. The project used EfficientNet B3 architecture and TinyImageNet as the source dataset and CIFAR-10 as the target dataset. The results showed that similar classes across the two datasets had strong influence on the performance. Removing the classes which had strong negative influence improved the prediction accuracy. However, there are challenges to deploying such systems in real-world applications, such as limited knowledge of the target dataset and compromising the performance of many other classes by removing some classes. Overall, the project's findings were in line with the results presented in the paper.