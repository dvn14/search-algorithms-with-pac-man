# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    A sample depth first search implementation is provided for you to help you understand how to interact with the problem.
    """
    
    mystack = util.Stack()
    startState = (problem.getStartState(), '', 0, [])
    mystack.push(startState)
    visited = set()
    while mystack :
        state = mystack.pop()
        node, action, cost, path = state
        if node not in visited :
            visited.add(node)
            if problem.isGoalState(node) :
                path = path + [(node, action)]
                break;
            succStates = problem.getSuccessors(node)
            for succState in succStates :
                succNode, succAction, succCost = succState
                newstate = (succNode, succAction, cost + succCost, path + [(node, action)])
                mystack.push(newstate)
    actions = [action[1] for action in path]
    del actions[0]
    return actions

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """    
    return 0

def modifiedHeuristic(node, problem, goal, fakeGoal, target, modify):
    if modify :
        if util.manhattanDistance(node, goal) < util.manhattanDistance(node, fakeGoal) :
            return 2*util.manhattanDistance(node, target)
    return util.manhattanDistance(node, problem.goal)

"""
COMP90054 AI Planning for Autonomy Assignment 1
Part 3 - Deceptive Path Planning Algorithm

Algorithm is referenced from the lecture notes: 2. Search Algorithms Slide. 32/50

aStarSearch is the support function to get the paths for the main function in searchAgents.py

node and state nomenclature is inverted.
"""

def aStarSearch(problem, heuristic=nullHeuristic, goal=None, fakeGoal=None, target=None, modify=False):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    myheap = util.PriorityQueue()
    startState = (problem.getStartState(), '', 0, [])
    node0, _, _, _ = startState
    h0 = modifiedHeuristic(node0, problem, goal, fakeGoal, target, modify)
    myheap.push(startState, h0)
    visited = set()
    while not myheap.isEmpty() :
        state = myheap.pop()
        node, action, cost, path = state
        if node not in visited :
            visited.add(node)
            if problem.isGoalState(node) :
                path = path + [(node, action)]
                break;
            succStates = problem.getSuccessors(node)
            for succState in succStates :
                succNode, succAction, succCost = succState
                newstate = (succNode, succAction, cost + succCost, path + [(node, action)])
                h = modifiedHeuristic(succNode, problem, goal, fakeGoal, target, modify)
                if h < float('inf') :
                    myheap.update(newstate, h)
    actions = [action[1] for action in path]
    del actions[0]
    return actions

"""
COMP90054 AI Planning for Autonomy Assignment 1
Part 1 - Enforced Hill Climbing

Algorithm is referenced from the lecture notes: 2. Search Algorithms Slide. 41-42/50

enforcedHillClimbing function is the main function
ehc_improve is the support function

node and state nomenclature is inverted.
"""

def ehc_improve(state0, problem):
    myqueue = util.Queue()
    myqueue.push(state0)
    visited = set()
    node0, _, _, _ = state0
    best_heuristic = util.manhattanDistance(node0, problem.goal)
    while not myqueue.isEmpty() :
        state = myqueue.pop()
        node, action, cost, path = state
        if node not in visited :
            visited.add(node)
            curr_heuristic = util.manhattanDistance(node, problem.goal)
            if curr_heuristic < best_heuristic :
                return state
            succStates = problem.getSuccessors(node)
            for succState in succStates :
                succNode, succAction, succCost = succState
                newstate = (succNode, succAction, cost + succCost, path + [(node, action)])
                myqueue.push(newstate)
    
    return 0

def enforcedHillClimbing(problem, heuristic=nullHeuristic):
    """COMP90054 your solution to part 1 here """
    mystack = util.Stack()      
    startState = (problem.getStartState(), '', 0, [])
    mystack.push(startState)
    while not mystack.isEmpty() :
        state = mystack.pop()
        node, action, cost, path = state
        if problem.isGoalState(node) :
            path = path + [(node, action)]
            break;
        new_state = ehc_improve(state, problem)
        mystack.push(new_state)
    actions = [action[1] for action in path]
    del actions[0]
    return actions
    
"""
COMP90054 AI Planning for Autonomy Assignment 1
Part 2 - Iterative Deepening A*

Algorithm is referenced from the page : https://en.wikipedia.org/wiki/Iterative_deepening_A* as instructed.

idaStarSearch function is the main function
ida_search is the support function

node and state nomenclature is inverted.
"""

def ida_search(visited_nodes, visited_states, g, threshold, problem):
    state0 = visited_states.pop()
    visited_states.push(state0)     # we just need the last element
    node0, action0, cost0, path0 = state0
    f = g + util.manhattanDistance(node0, problem.goal)
    if f > threshold :
        return (f, False, None)
    if problem.isGoalState(node0) :
        return (f, True, state0)
    lowerb = float('inf')
    succStates = problem.getSuccessors(node0)
    myheap = util.PriorityQueue()
    for succState in succStates :
        succNode, succAction, succCost = succState
        newstate = (succNode, succAction, cost0 + succCost, path0 + [(node0, action0)])
        priority = (cost0 + succCost) + util.manhattanDistance(succNode, problem.goal)
        myheap.update(newstate, priority)
    while not myheap.isEmpty() :
        state = myheap.pop()
        node, action, cost, path = state
        if node not in visited_nodes :
            visited_nodes.append(node)
            visited_states.push(state)
            new_threshold, goal_found, stateg = ida_search(visited_nodes, visited_states, cost, threshold, problem)
            if goal_found :
                return (new_threshold, True, stateg)
            if new_threshold < lowerb :
                lowerb = new_threshold
            visited_nodes.pop()
            visited_states.pop()
    return (lowerb, False, None)
    

def idaStarSearch(problem, heuristic=nullHeuristic):
    """COMP90054 your solution to part 2 here """
    startState = (problem.getStartState(), '', 0, [])
    node0, _, _, _ = startState
    threshold = util.manhattanDistance(node0, problem.goal)
    visited_states = util.Stack()
    visited_states.push(startState)
    visited_nodes = list()
    visited_nodes.append(node0)
    while True :
        new_threshold, goal_found, stateg = ida_search(visited_nodes, visited_states, 0, threshold, problem)
        if goal_found :
            break;
        if new_threshold == float('inf') :
            return 0
        threshold = new_threshold
    node, action, _, path = stateg
    path = path + [(node, action)]
    actions = [action[1] for action in path]
    del actions[0]
    return actions


                
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ehc = enforcedHillClimbing
ida = idaStarSearch