# Memristor-NN, Technion Project A.
- [Memristor-NN, Technion Project A.](#memristor-nn-technion-project-a)
  - [Background](#background)
  - [Project definition and goals](#project-definition-and-goals)
  - [Description of the algorithm](#description-of-the-algorithm)
    - [Manhattan Rule:](#manhattan-rule)
  - [Architectural design of the selected solution](#architectural-design-of-the-selected-solution)
    - [Y-Flash device](#y-flash-device)
  - [Status](#status)
    - [**Step 1**: Choosing dataset and simulating device behavior:](#step-1-choosing-dataset-and-simulating-device-behavior)
      - [Choosing Dataset:](#choosing-dataset)
    - [**Step 2**: Simulate single layered fully connected neural network and evaluating network performance](#step-2-simulate-single-layered-fully-connected-neural-network-and-evaluating-network-performance)
  - [Schedule for the remaining part](#schedule-for-the-remaining-part)
  - [Summary](#summary)
  - [Appendix:](#appendix)
    - [Appendix A: Iris Dataset](#appendix-a-iris-dataset)

## Background
Artificial neural networks (ANN) became a common solution for a wide variety of problems in many fields, such as control and pattern recognition. Many ANN solutions reached a hardware implementation phase, either commercial or with prototypes, aiming to accelerate its performance. Recent work has shown that hardware implementation, utilizing nanoscale devices, may increase the network performance dramatically, leaving far behind their digital and biological counterparts, and approaching the energy efficiency of the human brain. The background of these advantages is the fact that in analog circuits, the vector-matrix multiplication, the key operation of any neuromorphic network, is implemented on the physical level. The key component of such mixed-signal neuromorphic networks is a device with tunable conductance, essentially an analog nonvolatile memory cell, mimicking the biological synapse. There have been significant recent advances in the development of alternative nanoscale nonvolatile memory devices, such as phase change, ferroelectric, and magnetic memories. In particular, these emerging devices have already been used to demonstrate small neuromorphic networks. However, their fabrication technology is still in much need for improvement and not ready yet for the large-scale integration, which is necessary for practically valuable neuromorphic networks. This project investigates a network prototype based on mature technology of nonvolatile floating-gate memory cells.

## Project definition and goals
* Simulate single layered fully connected neural network and evaluating network performance for comparison.
* Simulate the same layer now using Manhattan Rule weights update and evaluating the rule influence on the network predictions.
* Simulating and implement the latter network now using weights being summed by positive and negative blocks as mentioned in the Manhattan Rule.
* Creating a Y-Flash neuron class to to simulate the network with small signal theory.
* Evaluating the performance under real world constraints such as: weight decrease in time or by temperature.
* Creating a Y-Flash neuron class to to simulate the network with large signal theory.
* Integration of the Y-Flash algorithm to lab equipment.
* To test what will happen if:
  * Neuron dies.


## Description of the algorithm
### Manhattan Rule:

## Architectural design of the selected solution
### Y-Flash device
* Physical behavior:
* Program and erase:

## Status
- [x] Choosing appropriate dataset for project - **Iris dataset**
- [x] Simulating device in Virtouso
- [ ] Simulate single layered fully connected neural network and evaluating network performance for comparison.
  - [ ] Code simple 1-layer neural network
  - [ ] Train and evaluate network performance
  - [ ] Compare to state of the art networks.

---
### **Step 1**: Choosing dataset and simulating device behavior:
#### Choosing Dataset:
 By hardware constraints we were bound to use a 8*12 Y-Flash device array. By this constraint and because we plan to implement the neuron weights with 2 Y-Flash devices one from positive and one for negative weight, we chose to work on the Iris data. Being a 4 input data set with 3 output predictions. A fitting choice since we have a 8 row array which computes into a 4 neuron layer since weights represented using 2 Devices.
 #### Simulating device: 

### **Step 2**: Simulate single layered fully connected neural network and evaluating network performance

 1. We chose to work with pytorch as our neural network platform because of the robustness and the available modification we can implement in order to simulate the Y-Flash device behavior.
 2. 

## Schedule for the remaining part

## Summary

## Appendix:
### Appendix A: Iris Dataset
From Wikipedia:

The **Iris flower** data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. Two of the three species were collected in the Gaspé Peninsula "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".

The data set consists of 50 samples from each of t**hree species** of Iris (Iris setosa, Iris virginica and Iris versicolor). **Four features** were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other


**Sample data:**

| Feature 1 | Feature 2 | Feature 3 | Feature 4 |
| --------- | --------- | --------- | --------- |
| 5.1       | 3.5       | 1.4       | 0.2       |
| 4.9       | 3.0       | 1.4       | 0.2       |
| 4.7       | 3.2       | 1.3       | 0.2       |
| 4.6       | 3.1       | 1.5       | 0.2       |
| 5.0       | 3.6       | 1.4       | 0.2       |
