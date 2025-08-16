
# Parameters in Machine Learning (Corrected & Complete)

## What are *parameters*?
In machine learning, **parameters** are the numeric values a model **learns from data** (e.g., weights and biases) so it can make predictions.

- They are updated during training to minimize a loss function.
- They differ by model type (linear models, neural nets, trees, etc.).

---

## Parameters vs. Hyperparameters
- **Parameters**: learned automatically (e.g., weights **w**, biases **b**).
- **Hyperparameters**: set by you before/around training (e.g., learning rate, number of layers, max depth).

---

## Core Model Examples (with correct equations)

### 1) Linear Regression
**Scalar form (2 features):**
$$
\hat{y} = w_1 x_1 + w_2 x_2 + b
$$

**Vector form (n features):**
$$
\hat{y} = \mathbf{w}^\top \mathbf{x} + b
$$

- **Parameters**: \(\mathbf{w} \in \mathbb{R}^n\), \(b \in \mathbb{R}\).  
- Trained by minimizing Mean Squared Error (MSE).

---

### 2) Logistic Regression (Binary Classification)
**Probability of class 1:**
$$
\hat{p}(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b), \quad \text{where } \sigma(z)=\frac{1}{1+e^{-z}}
$$

- **Decision**: \(\hat{y} = \mathbb{1}[\hat{p} \ge \tau]\) (often \(\tau = 0.5\)).  
- **Parameters**: \(\mathbf{w}, b\).  
- Trained by minimizing logistic loss (cross-entropy).

**Multiclass (Softmax) extension:**
$$
\hat{\mathbf{p}}(\mathbf{x}) = \text{softmax}(W\mathbf{x} + \mathbf{b}), \quad
\hat{p}_k = \frac{e^{(W\mathbf{x}+\mathbf{b})_k}}{\sum_{j} e^{(W\mathbf{x}+\mathbf{b})_j}}
$$

- **Parameters**: \(W \in \mathbb{R}^{K \times n}, \ \mathbf{b} \in \mathbb{R}^{K}\).

---

### 3) Neural Networks (Feedforward / MLP)
To avoid ambiguity, we separate **pre-activation** and **activation** explicitly.

**Single neuron:**
$$
z = \mathbf{w}^\top \mathbf{x} + b, \quad \hat{y} = f(z)
$$

**Layer \( \ell \) (vectorized):**
$$
\mathbf{z}^{[\ell]} = W^{[\ell]} \, \mathbf{h}^{[\ell-1]} + \mathbf{b}^{[\ell]}, 
\qquad
\mathbf{h}^{[\ell]} = f^{[\ell]}(\mathbf{z}^{[\ell]})
$$

- Here \( \mathbf{h}^{[0]} = \mathbf{x} \) (the input), and the final output is \( \mathbf{h}^{[L]} \).  
- **Parameters**: \( \{ W^{[\ell]}, \mathbf{b}^{[\ell]} \}_{\ell=1}^{L} \).  
- Shapes: if layer \( \ell \) has \( m_\ell \) units and layer \( \ell-1 \) has \( m_{\ell-1} \) units, then  
  \( W^{[\ell]} \in \mathbb{R}^{m_\ell \times m_{\ell-1}} \), \( \mathbf{b}^{[\ell]} \in \mathbb{R}^{m_\ell} \).

> **Why “\(y = f(Wx + b)\)” can be ambiguous:**  
> It mixes pre-activation and activation into one line and omits layer indices/shapes. The corrected, explicit form uses \(z\) then applies \(f\): \(z = Wx + b\), \(h = f(z)\), and includes layer indices for multi-layer nets.

**Convolutional layer (brief):**  
A convolution uses *kernels/filters* with parameters \(\mathbf{K}\). Each filter slides over the input and shares parameters spatially. Parameters are the filter coefficients and biases.

---

### 4) Decision Trees
- **Parameters** (learned structure): feature chosen at each split, threshold values, and the prediction values at leaves.  
- Not “weights” like neural nets; parameters are the **split rules** and **leaf outputs** learned from data.

---

### 5) k-Nearest Neighbors (k-NN)
- **No learned parameters** in the classic sense. It stores the training data and uses a hyperparameter \(k\) and a distance metric to make predictions.

---

## Real-Life Intuition

- **House prices (Linear Regression):** \( \mathbf{w} \) captures how much square footage, bedrooms, etc., contribute to price; \(b\) is the baseline.  
- **Spam filter (Logistic/NN):** weights on words like “FREE” push probability toward spam.  
- **Face recognition (CNN):** filters/weights detect edges → parts → faces.  
- **Self-driving (NN):** weights map sensor inputs to steering/braking decisions.

---

## “Trillion Parameters” in LLMs (ChatGPT-style models)
- In neural networks, **parameters = weights + biases**.  
- Saying a model has “~1 trillion parameters” means **it has that many learned values**.  
- More parameters ⇒ higher capacity (with enough data/regularization), but also higher compute/memory cost.

> **Note:** OpenAI has **not publicly disclosed exact parameter counts** for its newest models. “Trillion” is used to convey *scale*, not an official figure.

---

## Quick Glossary
- **Parameter**: a learned number inside the model (weight, bias).  
- **Hyperparameter**: a setting you choose (learning rate, layers, max depth).  
- **Activation \(f\)**: nonlinear function (ReLU, GELU, sigmoid, tanh).  
- **Pre-activation \(z\)**: linear combination before applying \(f\) (i.e., \(z = W h + b\)).

---

## TL;DR
- Use explicit layer notation for neural nets: \( \mathbf{z}^{[\ell]} = W^{[\ell]} \mathbf{h}^{[\ell-1]} + \mathbf{b}^{[\ell]} \), \( \mathbf{h}^{[\ell]} = f^{[\ell]}(\mathbf{z}^{[\ell]}) \).  
- Linear/logistic forms and shapes are now consistent and correct.
