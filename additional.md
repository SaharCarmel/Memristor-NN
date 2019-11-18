# limiting weights to memristor range
## basic limit [0,1]
This limit is very similar to pytorch's basic weight initialization. 

## biased narrow limit [1.45e3,1.45e3+1]
At first it may seem surprising that this experiment converged with a similar result. Very large weights with small learning rate and input data usually does not work well together. The model gave us good results is do to the fact that the layers are achieved as a difference between 2 layers, and thus the bias in weight values is (relatively) subtracted.

## biased wide limit [1/(1.45e6),1/(1.45e3)]
Here are model didn't converge well. We gave the weights a very wide range as opposed to the input and learning rate.

- [ ] We will now have to model the learning rate and input values to their as implemented in the physical device.
