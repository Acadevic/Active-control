# Automated modeling and control of steady state engineering processes using deep active learning


## Abstract
Active learning is a form of machine learning where the model selects the next appropriate action. It has the potential to automate both modeling and control of systems/processes, yet it is underutilized in engineering application. This is likely due to its non-transparent (black box) and automated characteristics, reducing the trust of engineering practitioners. This work present the topic in a manner that is familiar to practitioners, namely as an optimization problem using geometric constraints. During exploration the operating conditions can fluctuate changing from one extreme value to the next. A constraint is developed to address this issue.

## Keywords
Bayesian optimization, active learning, data-based modeling, steady state 



# Introduction

Active learning is a form of reinforcement learning that places an emphasis on the cost of sample acquisition, where modern reinforcement learning emphasizes stringing together sequence of decisions to escape local minima. For example [1] investigated “The design of experiment using reinforcement learning” by having a car escaping a bowl by driving around to build up enough momentum and catapult out. In contrast, active learning finds its roots in mining [2] where rock samples are extracted by drilling deep and expensive holes to find the optimum position for a mine. The goal is to find sample efficient data for modeling.  A large body of current work investigates active-learning for classification problems [3], by selecting the most informative samples to be labeled by a human oracle. According to [4] substantially less work considers regression.

This work investigates the use of active learning for control of industrial process. In this context it allows automated modeling and control of industrial processes. The authors anticipate that the adoption of this technique depends on the practitioner having confidence in the systems behavior. For this reason we present the systems behavior as an objective function constructed from a series of intuitive functions (constraints, transitions, exploration, and exploitation/control) which is more transparent but requires more design parameters.

This work also investigates a sampling constraint that limits extreme change of conditions between successive samples. This constraint is not applicable in simulation or clinical trials [5]–[7] but can affect efficiency and maintenance in engineering systems.

# Methodology

Running examples are used to illustrate the concepts here. Firstly a one dimensional problem presents how utility is constructed, using : (1) uncertainty sampling’s ability to reduce the number of trials, (2) range constraints to allow the practitioner to limit the systems operating conditions, and (3) a transition between exploration vs explorations using sigmoid functions.

A second multi-variate problem illustrates exploration constraints.

# Active learning

We present active-learning as a real-valued search problem. A system is sampled at particular operating conditions ( $y =f(x)+\epsilon$ ) . The data are used to train the model. Using the model the practitioner designs the acquisition function ( $u(x)$ ) to execute his/her intuition. The objective function, also referred to as acquisition or utility function is then searched to select the next sample. These acquisition functions are balance exploration and exploitation. Exploration considers acquiring informative samples, thereby efficiently modeling the system behavior and resembles optimal experimental design. Exploitation typically involves selecting samples that improve the current best, here we sampling at optimal conditions which resembles control.

![active learning](figs/flowchart.png)

Figure 1: Active learning involves constructing the utility function, searching it for the sample to acquire, then retraining the model.

This chapter looks at constructing the utility as an optimization search surface ( $u(x) \in R^1, x \in R^n$ ). It extends a more general form of Bayesian optimization ( ${u(x) = \eta + k\sigma}$) by providing explicit constraint terms as geometric functions. Where much of the previous work requires knowledge of Gaussian process regression, this work does not.  Several other forms are available in literature but are not as transparent [13]–[15], requiring deep knowledge of stochastic processes.

## Uncertainty sampling

Bayesian optimization uses a model to select the next point to sample automate experimental design. A number of acquisition functions exist, but here  uncertainty sampling is used. The intuition is to “measure at the point of highest uncertainty” and consequently highest information. 

<img src="figs/uncertainty_sampling.png">

Figure 2: The intuition behind uncertainty sampling is "acquiring samples at the point of highest uncertainty will improve the model". 

The figure shows sampling at the point of highest uncertainty (left) and how it tends to choose extreme values in sequence (right). In cases such as clinical trials this may not be an issue, but considering the input ( $x$ ) to be the temperature of a furnace we would prefer sequential values be "near". This issue is exacerbated by higher dimension inputs.

The avid reader will ask the question "are sequential predictions a better choice", but myopic sampling is investigated here as we may not always have a model than interpolates well enough.

Several modeling methods are able to predict the epistemic uncertainty (confidence interval) including Gaussian Process Regression, Ensemble, dropout, and Bayesian Neural Networks to name a few [8]–[12]. This work uses ensemble where multiple predictions quantify the mean and uncertainty ( $( \hat y,\sigma) = g(x)$ ).

# Multiplicative constraints

An issue that arises is that the sampling is not constrained  and may sample in undesirable states, far from our interest. Utility is introduced to accommodate constraints by quantifying the engineering practitioners requirements, desires, interests, etc. This can be thought of as a means of controlling or biasing the algorithms decision. 

Utility is the product of the uncertainty and the constraining function(s) ( $u(x) = \sigma(x) *c(x)$ ). The constraint is typically a step function ( $c(x) \in [0,1]$ ). The figure that follows illustrates this.


![Range constraint](figs/mc.png)

Figure 3: The construction of utility by the product of the uncertainty and constraint surfaces.

This acquisition function automates experimental design while respecting experimental constraints and has been shown to significantly reduce the number of experimental trials **[SELF REF]**. 

## Transition between exploration and exploitation

Once the model can predict the system response we would like to run the system at optimal conditions (control). We do this with the addition of an optimal response function ( $\eta$ ). The response is predicted by the model ( ${\eta(x) = h (\hat y})$ ) and so we require that the model predict the response correctly.

A term ( $w$ ) is introduce which can be loosely thought of as “the expected change in the model from the next sample”. When $w$ is low, there is little benefit in exploration, and exploitation/optimization can take precedence. Here we use the term to force the transition with a sigmoid function ( ${u(x) = c_x(x)[(1-w)*{\sigma (x)} +w{\eta(x)}] }$ ). This allows the practitioner to design a transition that varies from linear to discrete, ( ${w(z) \in (0,1)}$ ).

<img src="figs/transition.svg" alt="transition" align="center" />

Figure 4: Using a sigmoid as a transition function, the inflection point and slope will affect the transition between acquiring data to build the model (exploration) and controlling or optimizing the operation (exploitation).

Selecting to maximize the response ( ${\eta(x) = \hat y}$ ) the algorithm will now first explore to model the behavior, then optimize the operating conditions. The figure at the start of this chapter summarizes the process but a supplementary [GIF](figs/utility.gif) provides another perspective..

# Engineering application

This section explores a test case more indicative of engineering applications. The example test case is described and the  utility is constructed, illustrating some of the design choices regarding multivariate problems, vector scaling, and thrashing.



## Test case: Controlled temperature extruder

In this  case we consider model an extruder where the temperature of the material must be controlled. The goal is to select the element heat inputs $x= [Q_1 ,Q_2]$ such that the desired temperatures are reached $y = [T_2, T_3]$.

![extruder](figs/extruder.png	)

Figure 5: The extruder problem shown above, where the aim is to control the temperatures at specific points.

The control surface is described by the distance equation $\eta(x) = \frac{m}{m + ||s-g_{\mu}(x)||}$, where $m$ is the dimensionality of $y$. This will encourage the sampling at the operating conditions but requires the model correctly predict the response surface ( ${g{\mu} \simeq y }$ ).

![control surface](figs/control_surface.png)

Figure 6: The control surface for the extruder problem.



## Multi-variate decisions

Due to the limitations of multi-variate search techniques we require that the utility function return a single real-value. For this reason we need to norm the results such that ${||u(x)||  \in{ \mathbb{R}}^1}$ . In this work the $L_1$ norm was used.

For simplification and consistency we impose the rule that the utility is defined on the interval [0,1] ${(u, \sigma , c , w, \eta) \in [0,1] }$ . We ensure this by using the transformed variables. This has the added benefit that one dimension will not dominate another due to scale.![norm](figs/norm.png)

Figure 7: The transformed and norm variables used.

## Exploration constraints

As previously stated during exploration the system is expected to sample points of extreme values. In engineering applications this can be undesirable and dangerous.

A distance constraint encouraging consecutive samples be near to each other in the input space satisfies this goal $c_{exploration}=\frac{1}{1+k{slope}||x_{previous} - {x_{current}}||}$. . 

# Results and discussion

As predicted the algorithm first explored, gathering data to predict the response surface, then it optimizes the process. 

![utility surface](figs/utility_surface.png)

Figure 8: Shows the utility surface over time. Note how the surface looks different during the exploration phase. This [GIF](figs/utility_flow_original.gif) illustrates the change dynamically. 

A challenge of multi-variate problems is that the higher dimensional surfaces cannot always be visualized in familiar ways. Instead the inputs are shown for each iteration. The exploration and control phases can clearly be seen due to a almost discrete transition. Notice how fluctuations occurs during exploration, but the inputs settle at the desired outcome. The effect of the exploration is clearly visible in the figure (right). The $L_1$ norm was used encouraging changing a single dimension at a time. Similarly, the  $L_2$ could be used if the changing the values simultaneously may be preferred. Although the effect of norms may be known it is not clear when they should be appropriately applied to real-world systems.

![exploration](figs/exploration.png)

Figure 9: The exploration and exploitation phases can clearly be seen at 25 iterations. The effect of the exploration is clearly visible (right).

## Conclusion

This work presented active-learning as a method for automating both modeling and control of steady state systems. The method gives the practitioner a level of control by designing the transition, control surface, and constraints. The envisioned outcome is that active-learning be adopted in more industrial application by increasing practitioner trust through transparent methods and foreseeable outcomes.

This method is limited to steady state system. This assumption holds for a number of industrial systems but not for mechatronic systems such as the inverted pendulum. Hence a number of real-world applications  should be implemented to investigate where the boundary of appropriate use exists. A number of techniques need to be assimilated into frameworks for real world application, these techniques include steady-state detection, optimization techniques, splitting responsibilities between on-site real-time computation and cloud based computation.

# References

[1]	Christopher Gatti, *Springer Theses Recognizing Outstanding Ph.D. Research*. Accessed: Apr. 20, 2022. [Online]. Available: http://www.springer.com/series/8790

[2]	D. G. Krige, “A statistical approach to some mine valuations and allied problems at the Witwatersrand,” Master’s thesis, University of Witwatersrand, Witwatersrand, 1951.

[3]	P. Ren *et al.*, “A Survey of Deep Active Learning,” *ACM Comput. Surv.*, vol. 54, no. 9, Dec. 2022, doi: 10.1145/3472291.

[4]	R. Burbidge, J. J. Rowland, and R. D. King, “Active Learning for Regression Based on Query by Committee”.

[5]	A. P. Soleimany, A. Amini, S. Goldman, D. Rus, S. N. Bhatia, and C. W. Coley, “Evidential Deep Learning for Guided Molecular Property Prediction and Discovery,” *ACS Cent. Sci.*, vol. 7, no. 8, pp. 1356–1367, Aug. 2021, doi: 10.1021/acscentsci.1c00546.

[6]	D. Wu, C. T. Lin, and J. Huang, “Active learning for regression using greedy sampling,” *Inf. Sci.*, vol. 474, pp. 90–105, Feb. 2019, doi: 10.1016/J.INS.2018.09.060.

[7]	H. Yu and S. Kim, “Passive sampling for regression,” *Proc. - IEEE Int. Conf. Data Min. ICDM*, pp. 1151–1156, 2010, doi: 10.1109/ICDM.2010.9.

[8]	C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*, vol. 7, no. 5. 2000.

[9]	Y. Li *et al.*, “Deep Bayesian Gaussian processes for uncertainty estimation in electronic health records,” *Sci. Rep.*, vol. 11, no. 1, p. 20685, Dec. 2021, doi: 10.1038/s41598-021-00144-6.

[10]	Y. Gal and Z. Ghahramani, “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,” presented at the Proceedings of The 33rd International Conference on Machine Learning, Jun. 2015. [Online]. Available: https://proceedings.mlr.press/v48/gal16.html

[11]	T. Pearce, M. Zaki, A. Brintrup, and A. Neely, “High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach,” in *Proceedings of the 35th*, Feb. 2018. [Online]. Available: http://proceedings.mlr.press/v80/pearce18a.html

[12]	G. Yang, Y. Lin, and P. Bhattacharya, “A driver fatigue recognition model based on information fusion and dynamic Bayesian network,” *Inf. Sci.*, vol. 180, no. 10, pp. 1942–1954, 2010, doi: 10.1016/j.ins.2010.01.011.

[13]	J. Mockus, “Stochastic Models,” in *Bayesian Approach to Global Optimization: Theory and Applications*, J. Mockus, Ed. Dordrecht: Springer Netherlands, 1989, pp. 39–78. doi: 10.1007/978-94-009-0909-0_4.

[14]	N. Srinivas, A. Krause, S. M. Kakade, and M. Seeger, “Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design,” *IEEE Trans. Inf. Theory*, vol. 58, no. 5, pp. 3250–3265, May 2012, doi: 10.1109/TIT.2011.2182033.

[15]	H. J. Kushner, “A New Method of Locating the Maximum Point of an Arbitrary Multipeak Curve in the Presence of Noise,” *J. Basic Eng.*, vol. 86, no. 1, pp. 97–106, Mar. 1964, doi: 10.1115/1.3653121.
