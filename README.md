# Experiments used for the paper ~~accepted for~~ presented at [IJCNN2017](http://www.ijcnn.org/)
# Short-Term Plasticity in a Liquid State Machine Biomimetic Robot Arm Controller

## Abstract:
Biological neural networks are able to control limbs in different scenarios, with high precision and robustness. As neural networks in living beings communicate through spikes, modern neuromorphic systems try to mimic them making use of spike-based neuron models. Liquid State Machines (LSM), a special type of Reservoir Computing system made of spiking units, when it was first introduced, had plasticity on an external layer and also through Short-Term Plasticity (STP) within the reservoir itself. However, most neuromorphic hardware currently available does not implement both Short-Term Depression and Facilitation and some of them don't support STP at all. In this work we test the impact of STP in an experimental way using a 2 degrees of freedom simulated robotic arm controlled by an LSM. Four trajectories are learned and their reproduction analysed with Dynamic Time Warping accumulated cost as the benchmark. The results from two different set-ups showed the use of STP in the reservoir was not computationally cost-effective for this particular robotic task.


1) [Generation of trajectories](https://github.com/ricardodeazambuja/IJCNN2017/blob/master/2DofArm_simulation_data_generator-figures.ipynb)

2) [Simulated 2DoF arm](https://github.com/ricardodeazambuja/IJCNN2017/blob/master/2DofArm_simulation_data_generator_and_physics.ipynb)

3) [Generation of the training data](https://github.com/ricardodeazambuja/IJCNN2017/blob/master/2DofArm_simulation-Main.ipynb)

4) [Linear Regression - readout weights](https://github.com/ricardodeazambuja/IJCNN2017/blob/master/2DofArm_simulation_linear_regression.ipynb)

5) Testing:
- [Set A](https://github.com/ricardodeazambuja/IJCNN2017/blob/master/2DofArm_simulation_testing_learned_readouts-A-STP_ON.ipynb)
- [Set B](https://github.com/ricardodeazambuja/IJCNN2017/blob/master/2DofArm_simulation_testing_learned_readouts-B-STP_OFF.ipynb)
- [Set C](https://github.com/ricardodeazambuja/IJCNN2017/blob/master/2DofArm_simulation_testing_learned_readouts-C-STP_ON.ipynb)
- [Set D](https://github.com/ricardodeazambuja/IJCNN2017/blob/master/2DofArm_simulation_testing_learned_readouts-D-STP_OFF.ipynb)


## OBS:  
- [Results analysis 1](https://github.com/ricardodeazambuja/IJCNN2017/blob/master/___2DofArm_simulation_testing_analysis.ipynb)
- [Results analysis 2](https://github.com/ricardodeazambuja/IJCNN2017/blob/master/___2DofArm_simulation_testing_learned_readouts-analysis-metric-individual-sets.ipynb)
- [Results analysis 3](https://github.com/ricardodeazambuja/IJCNN2017/blob/master/___2DofArm_simulation_testing_learned_readouts-analysis.ipynb)
- [Internal structure visualisation](https://github.com/ricardodeazambuja/IJCNN2017/blob/master/2DofArm_simulation_3D_printing_of_liquid_structure.ipynb)
- [BEE SNN simulator](https://github.com/ricardodeazambuja/BEE)
- [Dynamic Time Warping Library](https://github.com/ricardodeazambuja/DTW)

## Preprint version:  
- [IJCNN2017_draft.pdf](https://github.com/ricardodeazambuja/IJCNN2017/raw/master/IJCNN2017_draft.pdf)

## Bibtex citation:
https://github.com/ricardodeazambuja/IJCNN2017/blob/master/de_azambuja_stp_2017.bib

## Final IEEE Xplore version:  
http://ieeexplore.ieee.org/document/7966283/

## Related works:
- [Graceful Degradation under Noise on Brain Inspired Robot Controllers](https://github.com/ricardodeazambuja/ICONIP2016)
- [Diverse, Noisy and Parallel: a New Spiking Neural Network Approach for Humanoid Robot Control](https://github.com/ricardodeazambuja/IJCNN2016)
- [Neurorobotic Simulations on the Degradation of Multiple Column Liquid State Machines](https://github.com/ricardodeazambuja/IJCNN2017-2)
- [Sensor Fusion Approach Using Liquid StateMachines for Positioning Control](https://github.com/ricardodeazambuja/I2MTC2017-LSMFusion)



