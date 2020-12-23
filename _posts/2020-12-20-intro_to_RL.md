---
layout: distill
title: Intro to Reinforcement Learning
description: Introduction to markov decision process, policies, and reinforcement learning.
date: 2020-12-20

authors:
  - name: Daqian Cheng
    affiliations:
      name: Robotics Institute, Carnegie Mellon University

---

# Markov Decision Process (MDP) and Reinforcement Learning (RL)

**RL vs Bandits**:  

| Similarities| Differences |
| --- | ------- |
| Possibly stochastic environment | Bandits: **single-step** decision making|
| Evaluative feedback | RL: **sequential** decision making|


## Markov Decision Process (MDP)

### MDP Concepts and Definitions

**Markov**: given the present state, the future and the past are independent.  
For MDP, action outcomes only depend on the current state: $$P(S_{t+1} = s'| S_t = s_t, A_t = a_t, S_{t-1} = s_{t-1}, A_{t-1} = a_{t-1}, \dots) = P(S_{t+1} = s'| S_t = s_t, A_t = a_t)$$

An **MDP** is defined by:
- A set of states $$s \in S$$  
- A set of actions $$a\in A$$  
- A transition function $$T(s, a, s')$$ of transition probabilities  
- A reward function $$r(s, a, s')$$  
- A start state $$s_0$$, or a distribution $$\mu_0(s)$$  

A **Trajectory**, a.k.a. an episode, is a sequence of states and actions.

### Reward and Goal

**Return** is the sum of rewards: $$R=\sum_{i=0}^\infty r_i$$

**Goal** is to find the actions that maximize the expected return: $$\max_{a_0,\dots,a_t} E[R]$$  
This goal does not capture:
- Constraints
- Maximize reward for **any** point along a trajectory
- Maximize reward at **some timestep t** along a trajectory

## Policies

For any MDP, a **policy** maps a state to an action or a distribution over actions: $$\pi: S\rightarrow A$$.  
**Optimal policy $$\pi^*$$**: maximizes expected return. Goal of RL is to find the optimal policy.

### Alternative Return

To avoid infinite return with infinite time steps, we can use
- Discounted reward: $$R(\pi) = \lim _{T\rightarrow \infty} E[\sum_i^T \gamma^i r_i]$$, where $$\gamma\in(0, 1)$$ imposes a sense of time horizon.
- Average reward: $$\bar{r}(\pi) = \lim _{T\rightarrow \infty} \frac{1}{T} E[\sum_i^T r_i]$$. This reward is better for cases of function approximation.

### Ranking Policies

Define **value function**: 

$$V_\pi(s) = \lim _{T\rightarrow \infty} E[\sum_i^T \gamma^i r_i|s_0=s, a_{0:t-1}\sim \pi]$$

To be better at a given state $$s$$: $$V_\pi(s) \geq V_{\pi'}(s)$$  
To be universally better: $$V_\pi(s) \geq V_{\pi'}(s)\ \forall s$$  
Policies have a **partial ordering**: it is possible to have two policies that $$V_\pi(s_1) \geq V_{\pi'}(s_1)$$ and $$V_\pi(s_2) \leq V_{\pi'}(s_2)$$, therefore neither is better than the other.

### Optimal Policy

A single best policy $$\pi^*$$ exists when there is no function approximation;  
A single best policy $$\pi^*$$ **may not** exist when there is function approximation;  
Why? Because two states with different optimal actions may be approximated with the same feature value, and in such cases there does not exist a single policy that satisfies both states.   
What can we do instead?
- Find optimal policy for a particular start-state $$s_0$$.
- Find optimal policy for a particular start-state distribution $$\mu_0(s)$$.
- Find optimal policy for the stationary distribution, which exists and does not depend on the start-state when the MDP is ergodic. This option does not reset the robot to a start-state, and is related to the average reward formulation.

$$d_\pi(s)=\lim_{T\rightarrow \infty}P(s_T=s|a_{0:t-1}\sim \pi)$$

For this course, it is assumed that the robot is reset to some initial state distribution $$\mu_0(s)$$, and we optimize for the expected discounted sum of rewards.

## Reinforcement Learning Methods

- **Model-based**
  - Sample efficient, best for learning in real-world
  - Sometimes give worse performance
  - Usually more interpretable
  - Easier to debug
  - Easier to encode prior information
  - Most generalizable to new environment
- **Q-function based**
  - In the middle of model-based and gradient-based methods.
- **Gradient-based**
  - Least efficient
  - Gives best performing policies
  - Fewest hyper-parameters to tune
  - Best for sim2real transfer

**Combinations** of the above methods are also possible, which often give the best performance with the fewest samples but require more tuning of hyper-parameters.


