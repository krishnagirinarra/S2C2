# Adaptive Coded Computation
This repository contains the software used for evaluations in Digital Ocean cloud environment.

This software is used in the following paper:

*Slack Squeeze Coded Computing for Adaptive Straggler Mitigation* </br>
**Krishna Giri Narra, Zhifeng Lin, Mehrdad Kiamari, Salman Avestimehr and Murali Annavaram** </br>
[arXiv:1904.07098](https://arxiv.org/abs/1904.07098)

## Disclaimer
This software is a proof-of-concept meant ONLY for performance evaluations of the Slack Squeeze Coded Computing(S2C2) framework.

## Running
### kubernetes related instructions
The workload to be evaluated is to be dockerized first and then run on a kubernetes cluster.

For bootstraping a k8s cluster, please refer to this link for more information -- https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/.

Once a k8s cluster is up and running, please refer to the README in the k8s/ folder for how to launch the workload to the cluster for evaluations. 

### docker related instructions
coded_computation_docker/codedComputation.Dockerfile is the Dockerfile used to dockerize the workloads.
coded_computation_docker/apps contains the all the workloads used for evaluations.
coded_computation_docker/apps/s2c2 contains the codes for S2C2 workload used for evaluations.
coded_computation_docker/apps/static contains the codes for conventional MDS workload used for evaluations.

## License
MIT License
