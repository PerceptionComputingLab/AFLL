# AFLL
***
Code for the paper Adjacency-aware Fuzzy Label Learning for Skin
Disease Diagnosis, TFS 2024

# Introduction
***
Automatic acne severity grading is crucial for the
accurate diagnosis and effective treatment of skin diseases. However, the acne severity grading process is often ambiguous due to
the similar appearance of acne with close severity, making it challenging to achieve reliable acne severity grading. Following the
idea of fuzzy logic for handling uncertainty in decision-making,
we transforms the acne severity grading task into a fuzzy label
learning problem, and propose a novel Adjacency-aware Fuzzy
Label Learning (AFLL) framework to handle uncertainties in this
task. The AFLL framework makes four significant contributions,
each demonstrated to be highly effective in extensive experiments.
First, we introduce a novel adjacency-aware decision sequence
generation method that enhances sequence tree construction by
reducing bias and improving discriminative power. Second, we
present a consistency-guided decision sequence prediction method
that mitigates error propagation in hierarchical decision-making
through a novel selective masking decision strategy. Third,
our proposed sequential conjoint distribution loss innovatively
captures the differences for both high and low fuzzy memberships
across the entire fuzzy label set while modeling the internal
temporal order among different acne severity labels with a cumulative distribution, leading to substantial improvements in fuzzy
label learning. Fourth, to the best of our knowledge, AFLL is the
first approach to explicitly address the challenge of distinguishing
adjacent categories in acne severity grading tasks. Experimental
results on the public ACNE04 dataset demonstrate that AFLL
significantly outperforms existing methods, establishing a new
state-of-the-art in acne severity grading.

![](img\main.jpg)

# Using the code
***
The code is stable while using Python 3.8.13, CUDA >= 11.6

- Clone this repository
```bath
git clone
cd AFLL
```
- To install all the dependencies using conda
```bath
cd env
conda env create -f env.yaml
conda activate AFLL
```
# Dataset
ACNE04 - [Link](https://github.com/xpwu95/ldl)

# Training and Validation
```bath
python main.py
```

# Citation
