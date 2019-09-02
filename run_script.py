from nn_experiments import ex
from nn_utils import visualize_digitization, Args

def main():
    ex.run(config_updates={'net': 'ref_net','digitizeInput':True}, options={'--name': 'phase_1'})
    ex.run(config_updates={'net': 'ref_net','digitizeInput':True}, options={'--name': 'phase_1'})
    ex.run(config_updates={'net': 'manhattan_net','digitizeInput':False}, options={'--name': 'phase_1'})
    ex.run(config_updates={'net': 'manhattan_net','digitizeInput':True}, options={'--name': 'phase_1'})


    args = Args('Parameters.yaml')
    visualize_digitization(args)

if __name__ == '__main__':
    main()