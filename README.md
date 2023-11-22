# EigenLayer-Federated-Learning
This repository is incomplete. It is supposed to be a rough interpretation of the architecture described below to implement a Federated 
Learning Network as an Actively Validated Service on EigenLayer. 
<br><br>
Currently Implemented:
- A CNN model using gorgonia training on MNIST dataset
- Operator training given base parameters and returning updated parameters
- Aggregator initialising a cnn model and sending tasks

## Abstract

Federated Learning currently faces two major issues 
- existence  of  malicious clients that decrease  the performance of the aggregated model, 
- exchange of local model parameters to compute federated averages can lead to leakage of private data. 

To counteract these issues I want to implement a group based FL model
(BGFLS) to easily backtrack malicious clients in a group of clients. The
ability identify bad actors can enable us to implement slashing conditions
through EigenLayer and establish crypto-economic security in the network.

### Problems with Byzantine-Resistant Schemes

There are several Byzantine resistance schemes like BGFLS, Krum and GeoMed. Each of them try to identify malicious clients and remove 
them from the network or try to minimize their effects on the aggregated parameters. The problem is that these methods are statistical 
and don’t eliminate the problem entirely.
<br>
Explained more about schemes like BGFLS and BPFL here: <br>
https://docs.google.com/document/d/1AWa8TTtIeR_mOKcSERe54G-BcIy_uEKcEIKtiAZ267s/edit?usp=sharing 

### Crypto Economic Security with EigenLayer

Allowing EigenLayer nodes to be clients in a federated learning network, you can enforce slashing conditions and establish economic security. 
This can eliminate malicious clients entirely from the network. <br>
Malicious clients can be identified using the average Euclidean distance between the parameter tensors (Krum)  or by checking similarity between 
each pair of parameter tensors (BGFLS). BGFLS also employs Bloom filters to increase efficiency of identification. <br>
These methods can identify outliers who’re bringing the model accuracy and hence can be slashed to disincentivize such behavior on the network. 


