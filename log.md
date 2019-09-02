# Meeting summaries and work log
### **Meeting #1:** *7/8/2019*
#### Meeting summary
#### Work log
#### Tasks
- [x] Reference MNIST network
- [x] yaml config
- [x] data loading
- [x] simulation logs
  - [x] Done with sacred/omniboard.
#### Questions
- [ ] Should we implement multi-layer modules? If so, should we implement activation layers?
- [ ]  Optimizer: currently implemtned constant step gradient descent. pytorch has an optimizer module (with better optimization algortihms..)  but we wouldn't be able to use the manahtan rule.

# Working log and summarize
## Reference Net
Reference net is a FC 2 layer network with:
* First layer: 783x800
* Second layer : 800x10

### Performance
| Metric            | Value |
| ----------------- | ----- |
| Accuracy          |       |
| Overfitting       |       |
| Convergence Time  |       |
| Power consumption |       |

### Plots

## Manahttan rule net
Manhattan net uses sign rule update scheme where weights are updated by the sign of the backpropagtion result

### Performance
| Metric            | Value |
| ----------------- | ----- |
| Accuracy          |       |
| Overfitting       |       |
| Convergence Time  |       |
| Power consumption |       |

### Plots