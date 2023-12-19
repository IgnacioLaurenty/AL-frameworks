# Active-Learning-Package

This package provides general tools for creating, studying and applying Deep Active Learning frameworks.

Ours toolsallow you to take into account an annotation cost, offer possibilities for partial annotations and the use of bayesian networks.

We do not provide any training method or networks, so you have to code them yourself, with the freedom to train on cpu, gpu or multi-gpu.

## Classic Frameworks

We provide well-known frameworks :
- Random Sampling
- Uncertainty Sampling (min-margin, max-entropy, least confidence)
- BALD
- BatchBALD
- Coreset (greedy)

We also provide another framework :
- ALPF (partial label active learning)


## Example notebook

We provide a notebook, giving examples of how you can use this package.
- numpy
- pytorch
- torchvision
- scikit-learn
- toma
- tqdm
- pickle

## Prerequisites




## References 

[1] A Sequential Algorithm for Training Text Classifiers, SIGIR, 1994

[2] Active Hidden Markov Models for Information Extraction, IDA, 2001

[3] Deep Bayesian Active Learning with Image Data, ICML, 2017

[4] Active Learning for Convolutional Neural Networks: A Core-Set Approach, ICLR, 2018

[5] Elementary applied statistics: for students in behavioral science. New
York: Wiley, 1965

[6] Active Learning with Partial Feedback, ICLR, 2019

[7] BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning, NIPS, 2019