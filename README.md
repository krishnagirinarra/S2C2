# Adaptive Coded Computation
This repository contains the software used for evaluations in Digital Ocean cloud environment.

# kubernetes related
The evaluated workload is to be dockerized first and then run on a kubernetes cluster.

For bootstraping a k8s cluster, please refer to this link for more information -- https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/.

Once a k8s cluster is up and running, please refer to the README in the k8s/ folder for how to launch the workload to the cluster for evaluations. 

# docker related
coded_computation_docker/codedComputation.Dockerfile is the Dockerfile used to dockerize the application.
coded_computation_docker/apps contains the all the workload used for evaluations.
coded_computation_docker/apps/s2c2 contains the codes for S2C2 workload used for evaluations.
coded_computation_docker/apps/static contains the codes for conventional MDS workload used for evaluations.

# License
MIT License
