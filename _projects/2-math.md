---
title: "Transformer Math"
excerpt: "Train transformer encoder-decoder models to solve \"math\" tasks. Focused on Integer Factorization > Pairwise Addition. <img src='/images/AdditionGeneralizationPlot.png'>"
collection: project
---

# Overview
Convert general math problems to sequences for transformer encoder decoder models. Encode the problem that should be solved, decode the answer to the problem.  
All models trained on one task individually, no multitask models. But that would be quite interesting.
* Factorization (4 --> 2x2)
* Pairwise Addition (45 + 202 --> 247)

# Why?
* Purpose of this project isn't to make a model that itself does something useful. It's more about exploring the trained models and how they generalize since their task is so well defined.
  * "In theory" the models could be useful. E.g. big number may be hard to factor, but if you predict posssible factors, easy to multiply and verify. And with beam search, it's easy to spit out a bunch of answers. Doesn't matter how many of them are wrong as long as one of them is right. But in practice, that's hard.
* "Literally infinite" perfectly labeled data. There are always more numbers.

# Results

## Pairwise Addition
This first model is trained on pairs of numbers [0,256] - with a 90/10 train/test split (pairs not individual numbers held out).
It gets 100% test accuracy on on the top beam.
[For some attention visualizations and exploring the embdddings, check out this notebooks](https://nbviewer.org/github/sims-s/neural-math/blob/main/notebooks/%5BPairwiseAddition%5D%20ModelExploration.ipynb).  Hard to make good sense of them.  
The most interesting plot:  
![](/images/AdditionGeneralizationPlot.png)  
This is model predicctions on a random 1% sampling of pairs of numbers between 0 and 400. The color of each circle is the index of the first beam that is the correct sum. Red means no correct beam in the top 5.    

The model stops generalizing once we hit 300. Which makes a lot of sense - the models were trained on numbers up to 256, so they've never encoutered a 3 in the hundreds digit. But they seem to generalize for [257, 299]

## Factorization
I explored this task much more. It's a lot more interesting.  

### Hyperparameters
The goal here is to find a reasonable set of hyperparameters to train a larger model on.  
For these hyperparameter experiments, my dataset was all integers $2\leq x\leq 2^{16} = 65536$.  
[All of the results are here.](https://nbviewer.org/github/sims-s/neural-math/blob/main/notebooks/%5BFactorization%5D%20HParamSearch.ipynb) I'm didn't write things about most of them, just the interesting stuff.

### Base
When encoding a number (input or output), we can encode it in any base we want, it doesn't need to be base 10. Training in larger bases leads to shorter sequences, which can be good. Especially for larger numbers.  
(**click to enlarge**)


[<img src="/images/FactorBaseComparison.png" width="2000">](/images/FactorBaseComparison.png)


* Both of these charts represent the same data, each row is a model trained with the shown hyperparameters (the ones not shown are all the same). The left is grouped by number of layers, and the right is sorted by base.  
* Base 30 encoding works better than the other bases I've tested here. I've found going higher didn't seem to have much benefit in earlier vesrions, so I didn't go larger. But could be worth revisiting.  
  
### Number of Encoder/Decoder layers
<!-- ![](/images/FactorEncoderLayerComparison.png) -->
[<img src="/images/FactorEncoderLayerComparison.png" width="1000">](/images/FactorEncoderLayerComparison.png)
* Interestingly, the larger stacks of models don't perform better. Maybe that's a function of $2^{16}$ not being very large.

### Attention Weight Initialization
In Pytorch, [attention weights are initialized here with `xavier_uniform_`](https://github.com/pytorch/pytorch/blob/b42cfcabc4e7216c5941ebab19c5000e5ecea4eb/torch/nn/modules/activation.py#L1048).  
This code creates a different weight initialization depending on whether or not q/k/v weights are grouped (one weight matrix $\left[3D\times D\right]$ size vs 3 $\left[D\times D\right]$s).  
With no other parameters specified, the initialization used for each parameter is Uniform$\left[-a,a]\right]$ where $a=\sqrt{\frac{6}{W.size(0) + W.size(1)}}$ for a given weight matrix W.  
So if we use the single large matrix, the weight initialization range will be decreased by a factor of $\sqrt{2}$. 

`Xavier_uniform_` takes in a `gain` parameter which is just multiplies everything inside the square root in the above equation. In my code, I've separated the q/k/v weights, but I use `gain=.5` in my initialization (making initialization equivelent to the chunked version) and find it makes a surprising difference. (left plot is test loss during training; right is out of sample - only numbers higher than the maximum training data)

![](/images/AttentionInitPlot.png)

Decreasing gain further didn't help, around .5 seemed to be the sweet spot. Though I didn't test too much further.

### Bigger Model
Now that I've figured out a hyperparameter setup, time to train a larger model. The model is 10 layers (even though fewer did better, this is a lot more data), trained in base 30 on all numbers (except a 20% test set) up to $2^{22}\approxeq 4.2 \text{ million}$  
[For all the results, see this notebook](https://nbviewer.org/github/sims-s/neural-math/blob/main/notebooks/%5BFactorization%5D%20ModelExplorations.ipynb)  

### Results & Thoughts
* 96.3% correct factorization rate; 99% correct product rate on the test set. Correct product means the numbers decoded produce the correct number, but not all of the numbers are prime.
  * The incorrect predictions have a very consistent behavior. They're "predicted as prime" - i.e. the model's prediction is the input itself. 
  * The model's training data contains prime numbers that follow that structure. So it follows it for hard numbers.
  * So potential room for improvment by changing the way the training data looks - that issue probably wouldn't happen if the model wasn't trained on primes.
    *  One simple idea is to get a giant list of primes (that's easy - I've toyed with the first 2 billion;  $\text{2 billion}^{th}\text{ prime}\approxeq 4.7 \times 10^{10}$) and just sample them. Pairwise or by trying to generate composites according to some distribution. Uniform composites is hard, I tried and got somewhat far. Kinda tricky 🤔
* We can also measure the generalization accuracy by looking at the 4096 next numbers after $2^{22}$. Using 10 beams, the model is 75% accurate.
* We can also measure the performance on the 4096 numbers after $2^{23}$. At 10 beams, the model is 17% accurate. Quite a drop! And remember the model likes regurgitating the input when it doesn't know what to do. And that *will* be right for primes in this dataset. I didn't measure this, but I suspect that means the model gets some "freebies" right. 

### Cosine Similarity Plot
Cosine similarity of embeddings:
![](/images/FactorizationCosineSim.png)
* The diagonal has been clipped to just above the maximum off diagonal value. The diagonal's always 1, but the other cosime sims are much smaller, so clipping it makes the plot easier to read.
* If you look closely, you can see things you'd expect:
  * (24, 5); (9,20) are dissimilar
  * evens are similar to evens; same with odds;
  * (25, 5) are similar

### Attention Visualization
[See it here](https://nbviewer.org/github/sims-s/neural-math/blob/main/notebooks/VisualizeAttention.ipynb).  
It's complicated, but a few things seem consistent:
* (EncEnc): During the first layer or two, the number tokens strongly attend to themselves across all heads.
* (EncEnc): After that first bit, the attention paterns are complex.
* (EncDec): During the first few layers, the model looks at all the previous tokens, but more so the start of sequence token.
* (EncDec): For the rest of the layers, the model really only uses the sos token. I wonder how much info we're really adding by doing this attention at every layer.
* (DecDec): All over the place. Tends to lean on the start of sequence token, but other tokens - particularly `*`, the multiplication symbol - also have high attention frequently enough.

