---
title: "Credit Assignment in the Brain"
categories:
  - review
tags:
  - essays
  - machine learning
  - backpropagation
  - neuroscience
  - 
---

 
# State representations are the basis for Reinforcement Learning

The world is full of complicated reward structures, but the brain can efficiently make decisions quickly and online. This efficiency is difficult to replicate using machine learning. In this essay, I will describe how the brain solves this problem and explore the learning processes that underlie it.

One such paradigm for understanding the brain’s decision-making processes is reinforcement learning. Reinforcement learning an adaptive learning process by which an agent uses its previous experience to improve the reward associated with outcomes in the future by making improved choices1. With appropriate computational scaling this method has been shown to be remarkably successful in performing complex tasks to higher than human level performance2.

Under this framework, the choices are made based on an estimate of the value associated with that choice according to some value function and making the set of decisions that maximise this reward metric into the future. Such value functions must evaluate the utility of a given action based on the positive and negative consequences of such an action further to internal states such as motivation. 
 
# The Basal Ganglia: Locating the Value Function in the Brain

There is evidence from a range of approaches that suggest that the basal ganglia is able to apply the computation required to implement the value function described in reinforcement learning. There is significant physiological evidence that in the basal ganglia, the dopamine signal acts as a teaching signal that can encode reward prediction error best described for a temporal difference learning algorithm3. Supporting this, there are computational models of how the internal circuits of the basal ganglia modify their synaptic connection strengths to update the value (i.e. payoffs and costs) that a particular action is associated with4. 

The view that the basal ganglia is important for computing value estimates is consistent with its anatomical description and cortical connectivity. The main output projection of the basal ganglia eventually projects back onto the cerebral cortex, via the substantia nigra and thalamus5. This is consistent with connectivity pattern that a value function would require if we imagine that our current understanding of the world around us is located in the cortex. Furthermore, this would suggest that the reward associated with an outcome is important in the updated model of the world that is being developed in the cortex, using reward prediction errors appropriately. This helps maintain a useful world model that can be used to guide further action. 

From this functional description of the basal ganglia, the value function clearly depends on (and reinforcement learning theory thus requires) a good state representation being present in the cortex, without which reasonable decisions cannot be made. 

## The cortex builds representations.
It is not a trivial task to build good state representations. For example, to maximise the reward associated with a kitchen, one requires a realisation that the white cabinets around them are not the result of being in a hospital or bathroom, but a kitchen. In this way, the exact state representation formed— given a complex sensory environment— is crucial for maximising the reward associated with an environment. This cognitive ability to “reason” about the world is consistent with case reports of people those with frontal lobe lesions, where such people are less able to perform well in a range of cognitive tasks6,7. If it is the indeed case that the cerebral cortex builds these state representations, we must ask (i) what type of state information does the cortex encode? And (ii), How does the brain update these state representations? The rest of this essay addresses these two questions.

## What types of information does the cortex encode?

To answer this question, let us first think of a useful way of representing the world. One such way of doing so is assuming that the current state of the world is entirely dependent on our understanding of the previous state of the world. Such a way of viewing state representation is known as a Markov decision process8,9.

It might seem limiting to have our state representation be dependant only on a previous state, however such a state representation could keep track of temporal states (e.g. “I have been waiting in the queue for 10 minutes”). In this way, it is possible for temporal limitations of MDPs to be overcome by generalising them to include hidden states. In fact, internally representing “hidden” information about the state the agent is in turns out to be very useful more generally.

Then, we must then ask ourselves the question what types of information we should keep in such a state representation. This turns out to be a difficult question to answer, especially in light of the realisation that some information critical for making a decision may not always be observable and must be inferred, whilst other perfectly observable things independent of progressing to the task are irrelevant. Furthermore, there are also internal states such as satiety, fatigue or motivation, that are likely influential in an agents state representation10 and state evaluation. The effect of our actions is also non-deterministic: when one looks out of the window, they must attempt to predict the future weather, whereas if one eats some food, they can predict with much more certainty the effect on their satiety. 

All these different modes of information must be captured and summarised into a state representation, such as a Markov model. Then the state representation can be sent to the basal ganglia, which can then send reward prediction errors back to the cortex to update the state representations9.

## State transitions

A way of capturing the probabilities of these state transitions in the cortex is in a structure known as a transition matrix, which— in its simplest form— is a table that captures the probability of transition from the state you are currently in, to any state in the state-transition matrix.

In reinforcement learning a distinction is made between “model free” and “model based” learning. The former suggests that transition probabilities are simply learnt, whereas the latter suggests that the agent can build a world model that allows it to understand the world well enough to choose a make decision that modify these transition probabilities. A model free approach has high computational efficiency, simply associating stimulus to response, whereas a model-based approach allows flexibility in decision making due to its implicit world mode. This is analogous to the prediction of the weather being model free and the effect of eating being model based. 

The Two-Step Problem11 attempts to disambiguate between the two decision making frameworks. However, experimental results show people appear to land somewhere between these two models behaviourally. It would be intuitive that our perception lies somewhere between this dichotomy, as the world we live in can only be partially understood. In fact, a more recent scientific interpretation suggests that humans could form a wide range of learning models to understand the Two-Step Problem, which can have varying interpretations based on how the task is explained to the participant12. 

A helpful resolution to this problem is the successor representation. The transition matrix can be thought of as a way of capturing the probability of being in all future states, from where you are now. From this information, it is possible to compute the probability of being in all future states at some point far in the future, by simply computing a constrained infinite sum of these transition probabilities. Such a transition matrix is known as the successor representation13,14. This view aligns with experimental evidence. It has striking similarities to the place field representation found in the hippocampus15, and the eigen-value decomposition of this matrix looks like the grid cells observed in the entorhinal cortex in animal models16; their traces are found in human entorhinal cortex17 even in non-spatial “navigation” tasks.  This interpretation provides an intermediate view between how a model based and model free implementation reinforcement learning might be done in the brain. This is because its model formulation based on both a model-based and model-free description makes a compromise between computational complexity and flexibility.

In the brain, there is a lot of evidence that the hippocampus is the primary place in which the brain summarises all the information that is important for the task into these state-transition matrices, for both spatial and non-spatial tasks. Upstream of the hippocampus, though, we might be interested in how this information is broken down. There is neural evidence that shows that the medial frontal cortex is able to keep track of a much more abstract notion of “task progress” in an animal model, where this animal model must navigate around four arbitrary, flexible waypoints a number of times before it get its satiety reward18. The animal model can break down the task into smaller parts by associating distinct neurons to particular features of the task’s subgoals (such as counting which cycle it is, or how close it is to a particular waypoint inside a cycle). 

Clearly being able to do this means that the animal model has built a useful state representation that is helpful for completing the task at hand. If it was keeping track of things uncorrelated with getting reward in the future, then it would likely face difficulty in achieving reward. Conversely, if it kept track of things that it could not control, it would not know how to make decisions that are rewarding. This is consistent with an intermediate state representation between model-free and model-based learning. 

There is an important cellular question about how these neural encodings of state are learnt in higher brain areas, how they are hierarchically organised, and when they are updated and represented. 

## How can the brain learn such a neural encoding of the states?

Understanding how reward prediction errors are integrated from the basal ganglia is question is experimentally challenging. We can used mathematical models of the cortex as they structural similarity to deep-learning models. The parameters of these models are analogous to the synaptic connection between neurons in the (layered) cortex. In this way the problem of understanding learning can be reframed as a problem of understanding how the parameters of these models change over their duration of learning. 

Unfortunately, this problem is also a challenging one. This is because we have hierarchically structured layers of processing in the cortex, and so the changing of a connection strength in one layer of the cortex can have large downstream effects in other parts of the cortex due to the highly interconnected nature of both the cortex and deep learning models. Knowing how to change these connect strengths is the essential challenge, and is known as credit assignment problem.

The state representations that are known to be found in the brain have important open questions: how these state representations generalise and what their basic units are how they can be compressed representations. However, addressing the credit assignment problem may be able to address the open questions presented by Whittington et al19. This is because they are all essentially questions of how weight representations are learnt and how they interact.

There is evidence that in simple deep learning models, there can do such a thing as keeping track of irrelevant variables (known as overfitting), however if you give a network the correct initial conditions (e.g. starting with small initial weights), the model is able to learn a “rich” representation of the world: that is, the model will only keep track of variables that are important for the task and will not keep track of irrelevant variables20. It is corroborated with experimental evidence that humans performing a visual task are able to abstract away unimportant features in fMRI signals21. This would be consistent with the El-Gaby’s results, where particular neurons represent important features of the task.

In Saxe et al.20, the training algorithm for updating weights is the error backpropagation algorithm22, which explicitly defines the loss function by defining an error metric and calculating the changes in parameters which best reduce this error metric and updates weights accordingly. It is not clear how the brain would be able to perform this task, as described in the backpropagation algorithm. This is because we know that neurons learn using local feedback between neurons 23. In particular, the training algorithm implicitly assumes that the required strength updates can be passed around the network and the synaptic strengths between layers of the cortex would be symmetrical. This is something that is highly unlikely, and is the crux of its biological implausibility24,25. This is known as the weight transport problem. 

Because of the success of the backpropagation algorithm however, people are interested in whether brains do in fact implement it. It follows that models of cortical learning that explicitly try to approximate the computations of backpropagation have been developed. An interesting example is the BurstPropagation Algorithm26. Here, the model is splits up the neuron into different compartments to perform different aspects of the required gradient computation and transportation. 

These types of model formulations would require an explicit representation of the loss function within the cortex, for which there is no such evidence, or would require creative non-biological work-arounds such as contrastive divergence27 to avoid such a representation. In Hinton’s paper27 one would have to run the forward model both with and without the training to find the prediction errors.

Instead of doing this, there exists another class of training algorithms which do not explicitly represent loss functions and only use local computation to minimise prediction errors: energy-based networks, in particular predictive coding networks 28. Due to their energy-based definition (where high prediction error in a particular layer cause the whole model to reduce its overall prediction errors) such networks can describe a much more biologically likely training algorithm compared with backpropagation. Under certain scenarios, the model can approximate the performance of backpropagation 29. However, instead of explicitly trying to approximate the weight updated that are implemented by backpropagation, the model chooses its own path. 

There is a natural probabilistic interpretation of predictive coding networks that arise from its formulation, namely that each layer predicts what the layers around it are doing which therefore can structure the task hierarchically and compositionally. This is in keeping with the El-Gaby’s paper, where individual cortical neurons are becoming tuned to different aspects of the task the animal model is performing. In this way, the cortex is ultimately building an internal model of the world that reflects reality by representing it in the firing rates of its neurons. 

The key distinction between predictive coding networks and backpropagation is that the model tries to obtain close to the prediction as possible simply by changing the neural activity of the network and only after this does the network changes the synaptic weights between layers. This is the opposite of what backpropagation does: here the weights are modified first and the impact of these weight modifications cause the neural activity in hidden layers to change30. 

This architectural difference between backpropagation and predictive coding leads to the model overcoming “catastrophic forgetting” (where the network trained using backpropagation can forgot an old association when learning a new one)31. The predictive coding network does not forget in this way as there are no associated local prediction errors. Therefore— since weight updates are proportional to local prediction errors— the weight updates to relevant neurons will be small. In backpropagation however, the errors backpropagated will overwrite any previous important associations. This overwriting arises from the non-local error metric that is used to update weights. 

In the task Saxe et al. presents, both predictive coding and backpropagation should have very similar results (as it is simple machine learning problem). However, as the learnt task becomes increasingly complicated, the structure and weight dynamics that occur should have different convergent values that are different between the models. 

In machine learning, one can define the concept of target alignment: the angular difference created between the weight updated prediction and target with the previous prediction generated by the learning algorithm (see Appendix)30. The target alignment is always closer using predictive coding compared with backpropagation, and it becomes increasingly divergent the greater the number of layers in the deep neural network. 

The predications of how the brain would update the estimates of states, is based on how it uses its prediction errors, generated in the basal ganglia. How the integration of these prediction errors is done leads to a range of outcomes. Appropriate synaptic weight updates are dependant both on the structure of the task and training algorithm. As the network becomes more complex (as a good model of the brain must be) there will be a greater range of weight dynamics that can perform the prescribed task. So, the algorithm used to train these models becomes increasingly important. 

# Conclusion
This essay has shown that there is a growing body of evidence that the cortex is the place where state representations of the world are formed in the brain. As our descriptions of these cells and their behaviour becomes more and more complete, we are forced to ask what a high-level mathematical description of synaptic weight updating happens in the cortex. 

There is no clear neuroscientific consensus on the algorithmic description of cortical learning. It may be that the learning dynamics described by different algorithms have different predictions and emergent behaviour in explaining the types of cortical cells we observe. Despite the practical superiority of the backpropagation algorithm, the possibility for such an algorithm to be implemented in the brain is uncertain. Meanwhile, energy-based algorithms, not only have a much more biological interpretation, but also have theoretical performance advantages. Thus, how the brain represents, and updates estimates of states are two interconnected problems that require further experimental and theoretical investigation for a consensus to be reached.
 
Appendix
Target Alignment Definition
 

Bibliography
1.	Sutton, R. S. & Barto, A. G. Reinforcement Learning, Second Edition: An Introduction. (MIT Press, 2018).
2.	Mnih, V. et al. Playing Atari with Deep Reinforcement Learning.
3.	Schultz, W., Dayan, P. & Montague, P. R. A neural substrate of prediction and reward. Science 275, 1593–1599 (1997).
4.	Bogacz, R. Theory of reinforcement learning and motivation in the basal ganglia. 174524 Preprint at https://doi.org/10.1101/174524 (2017).
5.	Redgrave, P., Prescott, T. J. & Gurney, K. The basal ganglia: a vertebrate solution to the selection problem? Neuroscience 89, 1009–1023 (1999).
6.	Collins, A. & Koechlin, E. Reasoning, Learning, and Creativity: Frontal Lobe Function and Human Decision-Making. PLoS Biol. 10, e1001293 (2012).
7.	Davis, S., Gupta, N., Samudra, M. & Javadekar, A. A case of frontal lobe syndrome. Ind. Psychiatry J. 30, S360–S361 (2021).
8.	Howard, R. A. Dynamic Programming and Markov Processes. viii, 136 (John Wiley, Oxford, England, 1960).
9.	Langdon, A. J., Song, M. & Niv, Y. Uncovering the ‘state’: tracing the hidden state representations that structure learning and decision-making. Behav. Processes 167, 103891 (2019).
10.	Berridge, K. C. Motivation concepts in behavioral neuroscience. Physiol. Behav. 81, 179–209 (2004).
11.	Daw, N. D., Gershman, S. J., Seymour, B., Dayan, P. & Dolan, R. J. Model-based influences on humans’ choices and striatal prediction errors. Neuron 69, 1204–1215 (2011).
12.	Feher da Silva, C. & Hare, T. A. Humans primarily use model-based inference in the two-stage task. Nat. Hum. Behav. 4, 1053–1066 (2020).
13.	Samuel J. Gershman. The Successor Representation: Its Computational Logic and Neural Substrates. J. Neurosci. 38, 7193 (2018).
14.	Stachenfeld, K. L., Botvinick, M. & Gershman, S. J. Design Principles of the Hippocampal Cognitive Map. in Advances in Neural Information Processing Systems vol. 27 (Curran Associates, Inc., 2014).
15.	O’Keefe, J. Place units in the hippocampus of the freely moving rat. Exp. Neurol. 51, 78–109 (1976).
16.	Moser, M.-B., Rowland, D. C. & Moser, E. I. Place Cells, Grid Cells, and Memory. Cold Spring Harb. Perspect. Biol. 7, a021808 (2015).
17.	Constantinescu, A. O., O’Reilly, J. X. & Behrens, T. E. J. Organizing Conceptual Knowledge in Humans with a Grid-like Code. Science 352, 1464–1468 (2016).
18.	El-Gaby, M. et al. A Cellular Basis for Mapping Behavioural Structure. 2023.11.04.565609 Preprint at https://doi.org/10.1101/2023.11.04.565609 (2023).
19.	Whittington, J. C. R., McCaffary, D., Bakermans, J. J. W. & Behrens, T. E. J. How to build a cognitive map. Nat. Neurosci. 25, 1257–1272 (2022).
20.	Saxe, A., Sodhani, S. & Lewallen, S. J. The Neural Race Reduction: Dynamics of Abstraction in Gated Networks. in Proceedings of the 39th International Conference on Machine Learning 19287–19309 (PMLR, 2022).
21.	Flesch, T., Saxe, A. & Summerfield, C. Continual task learning in natural and artificial agents. Trends Neurosci. 46, 199–210 (2023).
22.	Rumelhart, D. E., Hinton, G. E. & Williams, R. J. Learning representations by back-propagating errors. Nature 323, 533–536 (1986).
23.	Hebb, D. O. The Organization of Behavior; a Neuropsychological Theory. xix, 335 (Wiley, Oxford, England, 1949).
24.	Grossberg, S. Competitive learning: From interactive activation to adaptive resonance. Cogn. Sci. 11, 23–63 (1987).
25.	Liao, Q., Leibo, J. & Poggio, T. How Important Is Weight Symmetry in Backpropagation? Proc. AAAI Conf. Artif. Intell. 30, (2016).
26.	Greedy, W., Zhu, H. W., Pemberton, J., Mellor, J. & Ponte Costa, R. Single-phase deep learning in cortico-cortical networks. Adv. Neural Inf. Process. Syst. 35, 24213–24225 (2022).
27.	Hinton, G. E. Training Products of Experts by Minimizing Contrastive Divergence. Neural Comput. 14, 1771–1800 (2002).
28.	Rao, R. P. N. & Ballard, D. H. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nat. Neurosci. 2, 79–87 (1999).
29.	Whittington, J. C. R. & Bogacz, R. An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity. Neural Comput. 29, 1229–1262 (2017).
30.	Song, Y. et al. Inferring neural activity before plasticity as a foundation for learning beyond backpropagation. Nat. Neurosci. 27, 348–358 (2024).
31.	French, R. M. Catastrophic forgetting in connectionist networks. Trends Cogn. Sci. 3, 128–135 (1999).


