from source import solve_instances
from source.utils.ParametersParser import ParametersParser


def main():
    # Parse the parameters
    parameters = ParametersParser()

    # Solve the instances
    solve_instances.solve(parameters=parameters)


if __name__ == '__main__':
    main()
