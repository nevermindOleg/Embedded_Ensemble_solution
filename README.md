# Exact solution for the toy model of an Embedded Ensemble



## 1 Project background

Examples of embedded ensembles(EE) of neural networks are BatchEnsemble
arXiv:2002.06715 and dropout ensembles (with fixed set of masks). These en-
sembles can operate in individual regime, where the ensemble members are as
diverse as the members of independent ensemble, but the ensemble training is
harder, allowing only for a limited number of ensemble models. However, in the
limit of infinite network width, such ensembles can accomodate any number of
members and thus equivalent to the usual enesemble of independent networks.
The second EE operating regime is the collective regime, where ensemble mod-
els are inevitably correlated, but there is no limit on the number of ensemble
models, which can be trained simultaneously.

## 2 Project description

We propose to study the properties of EEs through the (potentially) exactly
solvable linear toy model. Although simple, we expect this toy model to cap-
ture qualitative behaviour of actual NNs, such as scalings of various quanti-
ties with network widthN and ensemble sizeM. Similar correspondece be-
tween linear and non-linear models was observed for usual (not ensembled) NNs
arXiv:1909.11304. The exact goals of the study are

- For the EE in individual regime, find dependence of its optimal sizeM∗
    on the network widthN.
- Examine transition from individual to collective regime by changing ini-
    tialization strategy of ensemble specific weights. If there is a sharp phase
    transition - find transition point.
- Experimentally check whether the behaviour found for the toy model is
    present in the ensembles of realistic NNs.
