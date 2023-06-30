#!/usr/bin/env python3

# This file accompanies the paper
#     "On a Computationally Ill-Behaved Bilevel Problem
#      with a Continuous and Nonconvex Lower Level"
# by Yasmine Beck, Dan Bienstock, Martin Schmidt, and Johannes Thuerauf (2023)
# (https://link.springer.com/article/10.1007/s10957-023-02238-9).

# Author: Yasmine Beck (yasmine.beck@uni-trier.de)

import gurobipy as gp
from gurobipy import GRB

def solve_lower_level(n, x, optimistic=True, tol=1e-10):
    # Input:
    # n -- Specifies the number of follower's variables.
    # x -- A feasible leader's decision.
    # optimistic -- Specifies the solution concept if the set of optimal
    #               follower's decisions is not a singleton.
    #               Default is True, i.e., the optimistic approach.
    #               If False is given, the pessimistic approach is used.

    # The function returns an optimal follower's decision for the given
    # leader's decision x as well as the corresponding optimal lower-level
    # objective function value.

    # Define lower and upper bounds for the follower's variables.
    lb = [0]*(n+1) + [-x[1]]
    ub = [GRB.INFINITY]*n + [x[0],x[1]]

    # Create model.
    model = gp.Model("lower-level problem")

    # Create follower's variables.
    y = model.addVars(n+2, vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name="y")

    # Set lower-level objective.
    model.setObjective(y[0] - y[n-1]*(x[0] + x[1] -y[n] - y[n+1]), GRB.MAXIMIZE)

    # Add lower-level constraints.
    model.addConstr(y[0] + y[n-1] == 0.5)
    for i in range(n-1):
        model.addConstr(y[i]**2 <= y[i+1])

    # Handle nonconvexities.
    model.params.NonConvex = 2

    # Increase numerical accuracy.
    model.params.NumericFocus = 3

    # Solve model.
    model.optimize()

    # Extract solution.
    sol = [y[idx].X for idx in range(n+2)]

    # If y_n = 0 holds, the variables y_n+1 and y_n+2 can be chosen
    # arbitrarily. Hence, we need to distinguish between the optimistic
    # and the pessimistic approach to bilevel optimization:
    if sol[n-1] < tol:
        [sol[n],sol[n+1]] = optimistic_vs_pessimistic(x, optimistic)

    return sol, model.objVal

def optimistic_vs_pessimistic(x, optimistic):
    # Create model.
    model = gp.Model()

    # Create follower's variables y_(n+1) and y_(n+2).
    y = model.addVars(2, vtype=GRB.CONTINUOUS, lb=[0,-x[1]], ub=[x[0],x[1]])

    if optimistic:
        # The follower wants to favor the leader, i.e., we maximize
        # the terms in the leader's objective function that depend
        # on the follower's variables.
        model.setObjective(-2*y[0] + y[1], GRB.MAXIMIZE)
    else:
        # The follower wants to adversely affect the leader, i.e., we
        # minimize the terms in the leader's objective function that
        # depend on the follower's variables.
        model.setObjective(-2*y[0] + y[1], GRB.MINIMIZE)

    # Solve model.
    model.optimize()
    return [y[0].X,y[1].X]

if __name__ == "__main__":
    # Specify the number of follower's variables.
    n = 6

    # Set leader's lower and upper variable bounds (lb and ub).
    # We require that 1 <= lb < ub holds.
    lb = [2,1]
    ub = [5,8]

    # Fix a feasibe leader's decision.
    x = ub

    # Solve the x-parameterized lower-level problem for the fixed
    # feasible leader's decision x.
    # The solution concept can be specified to be the optimistic or the
    # pessimistic approach to bilevel optimization by either setting
    # "optimistic" to True or False, respectively. The default is True.
    [y,follower_obj] = solve_lower_level(n, x, optimistic=False)

    print("The leader's objective function value for the fixed leader's decision %s is %.3f. \nThe follower's optimal reaction to the fixed leader's decision satisfies y_1 = %.3f, y_%d = %.3f, y_%d = %.3f, and y_%d = %.3f." %(x,x[0] - 2*y[n] + y[n+1],y[0],n,y[n-1],n+1,y[n],n+2,y[n+1]))
