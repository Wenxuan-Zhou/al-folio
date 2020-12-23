---
layout: distill
title: Perceptron & SVM
description: instructions to add a blog post on this website
date: 2020-12-21

authors:
  - name: Szuyu Lin
    affiliations:
      name: Robotics Institute, Carnegie Mellon University

bibliography: 2018-12-22-distill.bib

---

## Perceptron
The Perceptron can be viewed as a single-layer neural network. It is an online binary classification algorithm, which modifies its decision boundary after receiving each training instance.

### Perception's decision rule
For each input or training instance received, perception uses its decision rule to compute the dot product of the input $\vec{x}$ and current weights $\vec{w}$, then output the $sign (+/-)$ of the dot product.

$$ \hat{y}^{(t)} = sign(w^{(t-1)} \cdot x^{(t)}) $$

### Perception's update rule
In the online learning process, the perceptron then receives the true label of the training instance.

If the predicted sign is identical to the training instance, nothing happens. On the other hand, if the perceptron makes a mistake, the weights are updated with the update rule.

$$ w^{(t)} = w^{(t-1)} + y^{(t)} \cdot x^{(t)} \cdot 1[y^{(t)} \neq \hat{y}^{(t)}] $$

### Upper Bound
If the data is linearly separable, the perceptron algorithm is guaranteed to make a finite number of mistakes. 

To compute the mistake bound of perceptron, we define the "Potential Function" as following (the L2 norm of the weight vector):

$$ \Phi^{(t)} = \|{w^{(t)}}\|^2 = \sum(w_n^{(t)})^2 $$

If our learner makes a mistake at timestep $t$, the potential function becomes the following:

$$ 
\Phi^{(t)} = \left\Vert{w^{(t-1)} + y^{(t)}x^{(t)}} \right\Vert ^
 =\left\Vert{w^{(t)}}\right\Vert ^2 + \left\Vert x \right\Vert ^2 + 2y^{(t)} \langle w^{(t-1)}, x^{(t)} \rangle 
$$

How do we determine the sign of the $2y^{(t)} \langle w^{(t-1)}, x^{(t)} \rangle $ term, when the learner makes a mistake?

According to the decision rule, if the learner made a mistake, the prediction $\hat{y}^{(t)} = \langle w^{(t-1)}, x^{(t)} \rangle$ must have the opposite sign of the label $y^{(t)}$, 

the $2y^{(t)} \langle w^{(t-1)}, x^{(t)} \rangle $ term is negative, therefore the following inequality holds:

$$ \Phi^{(t)} \leq \left\Vert {w^{(t-1)}} \right\Vert ^2 + \left\Vert x^{(t)} \right\Vert ^2 $$

Let $R=\max_{t}\| x^{(t)}\| $ be the norm of input vector $x$, we can upper bound the potential function:

$$ \Phi^{(t)} \leq \left\Vert {w^{(t-1)}} \right\Vert ^2 + R^2 $$

On the other hand, it our learner does not make a mistake at timestep $t$, the update rule becomes the following:

$$w^{(t)} = w^{(t-1)}$$

And the potential function becomes:

$$\Phi^{(t)}=\left\Vert {w^{(t)}} \right\Vert ^2 = \left\Vert {w^{(t-1)}} \right\Vert ^2 $$

Combining both cases, our upper bound is:

$$ \Phi^{(t)} \leq \left\Vert {w^{(t-1)}} \right\Vert ^2 + R^2 $$

Starting from the base case, where $M^{(t)}$ is the total mistakes made at timestep $t$:

$$ \Phi^{(1)} \leq \left\Vert w^{(0)} \right\Vert ^2 + M^{(1)}R^2 $$

And the upper bound of the potential function after $T$ timesteps:

$$ \Phi^{(T)} \leq \left\Vert w^{(0)} \right\Vert ^2 + M^{(T)}R^2 $$

If we initialize $\left\Vert w^{(0)} \right\Vert $ to zero, we get:

$$ \Phi^{(T)} \leq M^{(T)}R^2 $$


### Lower Bound 

We assume the data to be linear separable here. Let $w^\star$ be the perfect classifier, and it is a unit vector, i.e. $\left\Vert w^\star \right\Vert = 1$ (assume exists).

The dot product $\langle w^\star, w^{(t-1)} \rangle$ will be the target we will derive the bounds for.

$$ \langle w^\star, w^{(t-1)} \rangle = \left\Vert w^\star \right\Vert \cdot \left\Vert w^{(T)} \right\Vert cos(\theta) \leq \left\Vert w^{(T)} \right\Vert $$

$cos(\theta) \leq 1$, so the upper bound on the dot product is: $\langle w^\star, w^{(t-1)} \rangle \leq \left\Vert w^{(T)} \right\Vert$.

For the lower bound derivation, we recall the update rule:

$$ w^{(t)} = w^{(t-1)} + y^{(t)} \cdot x^{(t)} \cdot 1[y^{(t)} \neq \hat{y}^{(t)}] $$

$$ \langle w^\star, w^{(t)} \rangle = \langle w^\star, w^{(t-1)} \rangle + y^{(t)} \langle w^\star, x^{(t)} \rangle \cdot 1[y^{(t)} \neq \hat{y}^{(t)}] $$

Since our $w^\star$ is a perfect classifier, the sign of $y^{(t)} \langle w^\star, x^{(t)} \rangle$ is guaranteed positive.

Now define the margin $\gamma = \min_{t} y_t \langle w^\star, x^{(t)} \rangle \geq 0$, we have the base case:

$$ \langle w^\star, w^{(1)} \rangle \geq \langle w^\star, w^{(0)} \rangle + \gamma \cdot M^{(1)}$$

$$ \langle w^\star, w^{(T)} \rangle \geq \langle w^\star, w^{(0)} \rangle + \gamma \cdot M^{(T)}$$

Initialize $w^{(0)} = 0$, we have the lower bound $ \langle w^\star, w^{(T)} \rangle \geq \gamma \cdot M^{(T)}$


### Combining both bounds

$$ \left\Vert w^{(t)} \right\Vert \geq \gamma \cdot M^{(T)} $$

Recall potential: $ \Phi^{(T)} = \left\Vert w^{(T)} \right\Vert ^2 $

We have the lower bound $ \Phi^{(T)} = \left\Vert w^{(T)} \right\Vert ^2 \geq (\gamma \cdot M^{(T)})^2, \Phi^{(T)} \geq (\gamma \cdot M^{(T)})^2 $

Combining with the upper bound $\Phi^{(T)} \leq M^{(T)} \cdot R^2 $, we have the mistake bound of perceptron:

$$ M^{(T)} \leq \frac{R^2}{\gamma^2} $$


### The margin

The margin is the distance between the decision boundary and the closest data point.

$$ \gamma = min_{t} y^{(t)} \langle w^\star, x^{(t)} \rangle > 0$$

The margin represents the degree of the separability of the data. If the margin is small, the data is harder to separate, therefore the learner makes more mistakes before finding the correct decition boundary.

For the norm $ R = max_{t} \left\Vert x^{(t)} \right\Vert $, if we have a large $R$, we will also have a larger mistake bound, and make more mistakes before finding the correct decision boundary.


### Incremental update SVM

Formula of a line: $w \cdot x + b = 0$, where $w, b$ are the weight and bias respectively.

Distance between line $w \cdot x + b = 0$ and origin: $\frac{b}{\left\Vert w \right\Vert}$

Distance between two parallel lines $w \cdot x + b = 0$ and $w \cdot x + (b+1) = 0$: $\frac{1}{\left\Vert w \right\Vert}$

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/dist.jpg">
    </div>
</div>


## Support Vector Machine

The best decision boundary is the one that maximizes the margin (i.e. distance to closest points), which is the most robust to purturbations in the inputs.

Solution for the best decision boundary:

for $y_{i}=+1, wx+b \geq \gamma$

for $y_{i}=-1, wx+b \leq -\gamma$

The margin here is $\frac{\gamma}{\left\Vert w \right\Vert}$, which is the distance between the decision boundary and the closest points to it, should be maximized.

The points on the lines $wx + b = \pm \gamma$ are called "support vectors".

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/svm.jpg">
    </div>
</div>


### Objective function

The object function of maximizing the margin is:
$$ max_{\gamma, w, b} \frac{\gamma}{\left\Vert w \right\Vert} \: s.t. wx_i+b \begin{cases} \geq \gamma, & y_i=+1 \\ \leq \gamma, & y_i=-1\end{cases} \, \forall i $$

If we set $\gamma = 1$, we can rewrite the objective and constraints:
$$ max_{w, b} \frac{1}{\left\Vert w \right\Vert} \: s.t. y_i(wx_i + b) \geq 1 \, \forall i $$

which is equivalent to:
$$ min_{w, b} \left\Vert w \right\Vert ^2 \: s.t. y_i(wx_i + b) \geq 1 \, \forall i $$

This is a convex quadratic programming (QP) problem, for which a unique solution exists.


### Soft margin

"Slack variables" make soft margin SVMs different from standard hard margin SVMs. 

Slack variables allow trade off between the margin and the object function, i.e. whether to linearly separate all data but have a very narrow margin, or maximize the margin allowing a few mistakes.

To include slack variables, we modify the previous constraint: $ min_{w, b} \left\Vert w \right\Vert ^2 s.t. y_i(wx_i + b) \geq 1 \, \forall i $

$$ min_{w, \zeta} \left\Vert w \right\Vert ^2 \: s.t. y_i(wx_i + b) \geq 1 - \zeta_i, \zeta_i \geq 0 \, \forall i $$

But if we have large $\zeta$'s, a large degree of mistakes would be allowed, though we might have a small objective but the SVM will learn nothing and have mistakes everywhere. Therefore we need a "regularization" term to prevent large $\zeta$'s.

$$ min_{w, \zeta} \left\Vert w \right\Vert ^2 + C \sum_i \zeta_i \: s.t. y_i(wx_i + b) \geq 1 - \zeta_i, \zeta_i \geq 0 \, \forall i $$

A large regularization term $C$ allows few mistakes and will produce a small margin. A small $C$, on the other hand, allows more mistakes and will result in a large margin.


## SGD for linear SVMs

Objective: $$ min_{w, \zeta} \left\Vert w \right\Vert ^2 + C \sum_i \zeta_i \: s.t. y_i(wx_i + b) \geq 1 - \zeta_i, \zeta_i \geq 0 \, \forall i $$

When we don't make a mistake, $y_i(wx_i + b) \geq 1$ holds, no need of slack, $\zeta_i=0$
 
When we make a mistake, $y_i(wx_i + b) < 1, \; \zeta_i = 1 - y_i(wx_i + b)$

Summarizing both cases we have: $\zeta_i = max(0, 1-y_i(wx_i + b)) $

And our new objective:

$$ min_{w, \zeta} \left\Vert w \right\Vert ^2 + C \sum_i max(0, 1-y_i(wx_i + b)) \: s.t. y_i(wx_i + b) \geq 1 - \zeta_i, \zeta_i \geq 0 \, \forall i $$


### Hinge loss and subgradients

$y_i(wx_i + b) = y_i \cdot \hat{y}_i$

Hinge loss $l_{hinge} = max(0, 1 - y_i \cdot \hat{y}_i)$

Soft SVM can be framed as a regularization convex loss optimization problem, so we can solve it with SGD!

SGD requires the gradient to be differentiable, however, the hinge loss is convex but not differentiable everywhere, therefore we have to take advantage of subgradients.

Use subgradients to optimize objective $min_{w} \left\Vert w \right\Vert ^2 + C \sum_i max(0, 1-y_m(w^T x_m))$: 

Subgradient of the objective:

$$
v = \begin{cases}
0 & y_m(w^T x_m) \geq 1 \\
-y_m x_m & y_m(w^T x_m) < 1
\end{cases}
$$

$v = -y_m x_m 1[y_m(w^T x_m) < 1]$


