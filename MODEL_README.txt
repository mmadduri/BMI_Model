MODEL README

2020_12_20
MathModels/BMIModel_MathModel_1D_NER.ipynb
- bug in the way that the perturbation was added and normalized to the brain parameters
- bug fixed to get simulations for the IEEE NER paper

2020_09_27
MathModels/BMIModel_MathModel_nD
--> Summer 2020
--> multi-dimensional, multi-neuron 
--> potential function = reach error + brain_regularization + decoder_regularization
--> the brain's parameters are updated  (W, b) with stochastic gradient descent
--> decoder's parameters are updated with SGD

TESTS: 

----------------------------------------------------------------------------------------
2020_09_25
MathModels/BMIModel_MathModel_1D
--> Summer 2020
--> 1 neuron, 1D (goes from -1 to 1)
--> potential function/system cost = reach error + brain_regularization + decoder_regularization
--> brain's parameters (W) are updated with sgd
--> decoder's parameters are updated with SGD

TESTS:
- stationary point of K = sqrt(-sigma_k/tau^s + 1)
- stationary point of W = sqrt(-sigma_w/tau^s + 1)


----------------------------------------------------------------------------------------
Earlier Models:

OldModels/BMI_AdaptiveDecoder
--> original work for Winter 2020, Spring 2020
--> multi-dimensional, multi-neuron
--> originally the system cost = reach error (only)
--> based closely on Heliot, et al 2010 paper and it's model


OldModels/BMI_firingrates_perturbations_Scalar
--> 1D
--> changed to match the 20200810-learning-rate-bmi calculations
--> potential function/system cost = reach error + brain_regularization + decoder_regularization
--> brain cost = reach error + brain_regularization
--> decoder cost = reach error + decoder_regularization


