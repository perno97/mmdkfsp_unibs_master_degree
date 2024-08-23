import argparse
from typing import Sequence

# Example of launch command:
# python source/solve_instances.py -t 60 -i 50 -ni 10 -k 5 -fa 10 -ks 10 -is 10 -mi 10 -r 3 -it 30 -wr -wa -wi -wk -d

DEFAULT_TIME = 60
DEFAULT_ITERATIONS = 0
DEFAULT_NO_IMPROVEMENT = 150
DEFAULT_K_MAX = 2  # 7
DEFAULT_FAMILY_ADDITION = 30  # 1 # Must be low, lower than the number of families, because otherwise it will try to add always the same families
DEFAULT_KNAPSACK_SELECTION = 20  # 3 #1
DEFAULT_ITEM_SELECTION = 10  # 5
DEFAULT_MOVE_ITEM = 100  # 5 # It will be multiplied by the number of items to move, so it must be low
DEFAULT_RESTART = 80  # 5 # Must decrease if the execution is slower (less iterations), otherwise it will restart a few times
DEFAULT_WEIGHTS_REMOVING = True
DEFAULT_WEIGHTS_ADDING = True
DEFAULT_WEIGHTS_ITEMS = True
DEFAULT_WEIGHTS_KNAPSACKS = True
DEFAULT_DEBUG = False
DEFAULT_REMOVE_COUNTER_WEIGHTS_REMOVING = False
DEFAULT_SELECTION_COUNTER_WEIGHTS_REMOVING = False
DEFAULT_SELECTION_COUNTER_WEIGHTS_ADDING = False
DEFAULT_REMOVE_COUNTER_WEIGHTS_ADDING = False


class ParametersParser:
     """
          Class to parse the parameters from the command line.
     """
     def __init__(self, args=None):
          args = self.parse_args(args)
          self.time: int = args.time
          self.iterations: int = args.iterations
          self.no_improvement: int = args.no_improvement
          self.k_max: int = args.k_max
          self.family_addition: int = args.family_addition
          self.knapsack_selection: int = args.knapsack_selection
          self.item_selection: int = args.item_selection
          self.move_item: int = args.move_item
          self.restart: int = args.restart
          self.weights_removing: bool = args.weights_removing
          self.weights_adding: bool = args.weights_adding
          self.weights_items: bool = args.weights_items
          self.weights_knapsacks: bool = args.weights_knapsacks
          self.debug: bool = args.debug

          self.remove_counter_weights_removing = args.remove_counter_weights_removing
          self.selection_counter_weights_removing = args.selection_counter_weights_removing
          self.selection_counter_weights_adding = args.selection_counter_weights_adding
          self.remove_counter_weights_adding = args.remove_counter_weights_adding

     def parse_args(self, args: Sequence[str] | None) -> argparse.Namespace:
          parser = argparse.ArgumentParser()
          parser.add_argument('-t', '--time', type=int,
                              help='The time limit in seconds, 0 for no limit. It\'s '
                                   'checked at the beginning of each iteration.',
                              default=DEFAULT_TIME)
          parser.add_argument('-i', '--iterations', type=int,
                              help='The number of iterations limit, 0 for no limit.',
                              default=DEFAULT_ITERATIONS)
          parser.add_argument('-ni', '--no-improvement', type=int,
                              help='The number of iterations without improvement limit,'
                                   '0 for no limit. It must be less than the number of '
                                   'iterations limit.',
                              default=DEFAULT_NO_IMPROVEMENT)
          parser.add_argument('-k', '--k-max', type=int,
                              help='The maximum neighborhood structure value. This is '
                                   'proportional to the neighborhood size, so with a bigger '
                                   'k value the algorithm explores more neighbors.',
                              default=DEFAULT_K_MAX)
          parser.add_argument('-fa', '--family-addition', type=int,
                              help='The maximum number of unfeasible tries to add '
                                   'a family. The 0 value means no limit.',
                              default=DEFAULT_FAMILY_ADDITION)
          parser.add_argument('-ks', '--knapsack-selection', type=int,
                              help='The maximum number of unfeasible tries to '
                                   'select a knapsack when trying to add a family. '
                                   'The 0 value means no limit.',
                              default=DEFAULT_KNAPSACK_SELECTION)
          parser.add_argument('-is', '--item-selection', type=int,
                              help='The maximum number of unfeasible tries '
                                   'to select the items to move. The `0` value '
                                   'means no limit.',
                              default=DEFAULT_ITEM_SELECTION)
          parser.add_argument('-mi', '--move-item', type=int,
                              help='The maximum number of unfeasible tries '
                                   'to move the selected items to a random knapsack '
                                   'before trying a different selection of items '
                                   '(if -is limit hasn\'t been reached). The 0 value '
                                   'means no limit.',
                              default=DEFAULT_MOVE_ITEM)
          parser.add_argument('-r', '--restart', type=int,
                              help='The number of iterations without improvement '
                                   'before restarting from the next initial solution '
                                   'built. The 0 value means no restart.',
                              default=DEFAULT_RESTART)
          parser.add_argument('-wr', '--weights-removing', action='store_false',
                              help='If set, disables weights when selecting '
                                   'a family to remove (more weight to families '
                                   'with higher additions, for example).',
                              default=DEFAULT_WEIGHTS_REMOVING)
          parser.add_argument('-wa', '--weights-adding', action='store_false',
                              help='If set, disables weights when selecting '
                                   'a family to add (less weight to families with '
                                   'higher additions, for example).',
                              default=DEFAULT_WEIGHTS_ADDING)
          parser.add_argument('-wi', '--weights-items', action='store_false',
                              help='If set, disables weights when selecting '
                                   'the items to move (less weight to items '
                                   'selected many times).',
                              default=DEFAULT_WEIGHTS_ITEMS)
          parser.add_argument('-wk', '--weights-knapsacks', action='store_false',
                              help='If set, disables weights when selecting '
                                   'the knapsack in which to move an item. (less '
                                   'weight to knapsacks selected many times).',
                              default=DEFAULT_WEIGHTS_KNAPSACKS)
          parser.add_argument('-d', '--debug', action='store_true',
                              help='If set, enables debug mode, so the '
                                   'algorithm prints some more information about '
                                   'the execution.',
                              default=DEFAULT_DEBUG)
          parser.add_argument('-rwr', '--remove-counter-weights-removing', action='store_true',
                              help='If set, use the remove family counter when '
                                   'calculating the weights when removing a family',
                              default=DEFAULT_REMOVE_COUNTER_WEIGHTS_REMOVING)
          parser.add_argument('-swr', '--selection-counter-weights-removing', action='store_true',
                              help='If set, use the selection counter when '
                                   'calculating the weights when removing a family.',
                              default=DEFAULT_SELECTION_COUNTER_WEIGHTS_REMOVING)
          parser.add_argument('-swa', '--selection-counter-weights-adding', action='store_true',
                              help='If set, use the selection counter when '
                                   'calculating the weights when adding a family.',
                              default=DEFAULT_SELECTION_COUNTER_WEIGHTS_ADDING)
          parser.add_argument('-rwa', '--remove-counter-weights-adding', action='store_true',
                              help='If set, use the remove family counter '
                                   'when calculating the weights when adding a family.',
                              default=DEFAULT_REMOVE_COUNTER_WEIGHTS_ADDING)
          if args is None:
               return parser.parse_args()
          else:
               return parser.parse_args(args)
