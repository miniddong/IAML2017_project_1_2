import train
from constant import Constant
from parameters import Parameters

project_constant = Constant.Project2
data_constant = Constant.Data.ChromaStftHop512
experiment_parameters = Parameters.BidirectionalAttention.Experiment1
is_training = True

def main():
    if is_training:
        train.train(project_constant, data_constant, experiment_parameters)

main()