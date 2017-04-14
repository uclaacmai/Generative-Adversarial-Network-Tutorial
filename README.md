# Generative-Adversarial-Network-Tutorial

Generative adversarial networks (GANs) have been one of the hottest topics in deep learning. From a high level, GANs are composed of two components, a generator and a discriminator. The discriminative model has the task of determining whether a given image looks natural (an image from the dataset) or looks like it has been artificially created. The task of the generator is to create natural looking images that are similar to the original data distribution. The analogy used in the paper is that the generative model is like “a team of counterfeiters, trying to produce and use fake currency” while the discriminative model is like “the police, trying to detect the counterfeit currency”. The generator is trying to fool the discriminator while the discriminator is trying to not get fooled by the generator. As the models train through alternating optimization, both methods are improved until a point where the “counterfeits are indistinguishable from the genuine articles”. 

The tutorial is written in Python, with the Tensorflow library, so it would be good to have familiarity with Tensorflow before taking a look at this tutorial. 

# How to Use Jupyter Notebooks



# More Resources

* The original [paper](https://arxiv.org/pdf/1406.2661.pdf) written by Ian Goodfellow in 2014. 
* Siraj Raval's [video tutorial](https://www.youtube.com/watch?v=deyOX6Mt_As) on GANs (Really fun video)
* Ian Godfellow's [keynote](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Generative-Adversarial-Networks) on GANs (More of a technical video)
* Brandon Amos's image completion [blog post](https://bamos.github.io/2016/08/09/deep-completion/)
* [Blog post](https://medium.com/@ageitgey/abusing-generative-adversarial-networks-to-make-8-bit-pixel-art-e45d9b96cee7) on using GANs in video games. 
* Andrej Karpathy's [blog post](http://cs.stanford.edu/people/karpathy/gan/) with GAN visualizations.
* Adit Deshpande's [blog post](https://adeshpande3.github.io/adeshpande3.github.io/Deep-Learning-Research-Review-Week-1-Generative-Adversarial-Nets) on GANs.
