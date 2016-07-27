Re-implementation of the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by 
[Kaiming He](https://github.com/KaimingHe) et Al. using Google's [TensorFlow](https://www.tensorflow.org/) library.

I implemented this paper for my Bachelor's thesis at Technische Universität München, 
[Chair of Scientific Computing](http://www5.in.tum.de/wiki/index.php/Home).

# Results

You can read all about my experiments and results in my thesis (located in the `thesis` folder). Here's a short summary 
of the most important findings.

Due to hardware restrictions I was only able to test the models ResNet-20, ResNet-32, ResNet-44 and ResNet-56 on the 
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. This is what I got, compared to the results from the 
original paper:

## My results
![My results on CIFAR-10][my_results]

## Results from the original paper
![Results on CIFAR-10 from the original paper][original_results]

## Comparison
| **Model** | **my test error** | **test error in original paper** |
|:---------:|:-----------------:|:--------------------------------:|
| ResNet-20 | 8.28%             | 8.75%                            |
| ResNet-32 | 8.03%             | 7.51%                            |
| ResNet-44 | **7.13**%         | 7.17%                            |
| ResNet-56 | 7.40%             | **6.97**%                        |

# Usage

If you'd like to run your own experiments with my implementation, take a look into the `shellscripts` folder for 
examples how to run training and evaluation on different models. All configurable parameters can be found in the file
`config.py`.

On `master` there also exist stubs for training and evaluating the models on other datasets, namely the 
[ImageNet](http://www.image-net.org/) dataset and the image data of the 
[Yelp Restaurant Photo Classification](https://www.kaggle.com/c/yelp-restaurant-photo-classification) challenge. For the
latter there are working implementations in the branches `yelp`, `yelp-evaluation` and `yelp-testing`. However, I highly
discourage using them since the code is quite messy.

[my_results]: thesis/my_results.png "My results on CIFAR-10"
[original_results]: thesis/original_results.png "Results on CIFAR-10 from the original paper"