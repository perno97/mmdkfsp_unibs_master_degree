import traceback
from source import solve_instances
from source.utils.ParametersParser import ParametersParser


def main():
    """
    The starting point of the program.
    It takes the parameters from the command line and calls the script "solve_instances.py".
    """    
    # Parse the parameters
    parameters = ParametersParser()
    # Solve the instances
    solve_instances.solve(parameters=parameters)


if __name__ == '__main__':
    main()
