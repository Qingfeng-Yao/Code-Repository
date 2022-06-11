from utils import *

class Problem:
    # The abstract class for a formal problem.
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        # The result would typically be a list
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def goal_test(self, state):
        # Return True if the state is a goal
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        # assuming cost c to get up to state1
        # The default method costs 1 for every step in the path
        return c + 1

    def value(self, state):
        # For optimization problems, each state has a value.
        raise NotImplementedError

class Node:
    # A node in a search tree.
    def __init__(self, state, parent=None, action=None, path_cost=0):
        # Create a search tree Node, derived from a parent by an action.
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        # List the nodes reachable in one step from this node.
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        # Return the sequence of actions to go from the root to this node.
        return [node.action for node in self.path()[1:]]

    def path(self):
        # Return a list of nodes forming the path from the root to this node.
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state stored in the node 
        # instead of the node object itself 
        # to quickly search a node with the same state in a Hash Table
        return hash(self.state)

# ______________________________________________________________________________

class SimpleProblemSolvingAgentProgram:
    # Abstract framework for a problem-solving agent.
    def __init__(self, initial_state=None):
        # seq is the list of actions required to get to a particular state from the initial state(root).
        self.state = initial_state
        self.seq = []

    def __call__(self, percept):
        # Formulate a goal and problem, then search for a sequence of actions to solve it.
        self.state = self.update_state(self.state, percept)
        if not self.seq:
            goal = self.formulate_goal(self.state)
            problem = self.formulate_problem(self.state, goal)
            self.seq = self.search(problem)
            if not self.seq:
                return None
        return self.seq.pop(0)

    def update_state(self, state, percept):
        raise NotImplementedError

    def formulate_goal(self, state):
        raise NotImplementedError

    def formulate_problem(self, state, goal):
        raise NotImplementedError

    def search(self, problem):
        raise NotImplementedError

