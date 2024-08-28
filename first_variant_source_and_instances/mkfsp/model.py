from __future__ import annotations

from typing import NamedTuple

from gurobipy import GRB, Model, quicksum, tupledict, Env

from .instance import Instance


class ModelVars(NamedTuple):
    """The Gurobi model of a MKFSP instance and its varibles."""

    model: Model
    xvars: tupledict
    yvars: tupledict
    zvars: tupledict
    svars: tupledict


def build_model(instance: Instance, disableLogs: bool = False) -> tuple[Model, ModelVars]:
    """Construct a Gurobi model for the given MKFSP instance.

    Args:
        instance (mkfsp.instance.Instance): an instance of the MKFSP problem

    Returns:
        ModelVars: the Gurobi model and its variables divided by family.
    """

    EQ = GRB.EQUAL
    LEQ = GRB.LESS_EQUAL

    n_items = instance.n_items
    n_families = instance.n_families
    n_knapsacks = instance.n_knapsacks
    profits = instance.profits
    penalties = (-v for v in instance.penalties)
    first_items = instance.first_items
    items = instance.items
    knapsacks = instance.knapsacks

    with Env(empty=True) as env:
        if disableLogs:
            env.setParam('OutputFlag', 0)
        env.start()
        model = Model(instance.id, env=env)
        xvars = model.addVars(n_families, obj=profits, vtype='B', name='x')
        yvars = model.addVars(n_items, n_knapsacks, obj=0, vtype='B', name='y')
        zvars = model.addVars(n_families, n_knapsacks, obj=0, vtype='B', name='z')
        # svars = model.addVars(n_families, obj=penalties, vtype='I', name='s')
        uvars = model.addVars(n_families, obj=penalties, vtype='B', name='u')
        model.modelSense = GRB.MAXIMIZE

        _addLConstr = model.addLConstr
        for j, first_item in enumerate(first_items):
            end_item = first_items[l] if (l := j+1) < n_families else n_items

            # Add family integrity constraints
            x_j = xvars[j]
            for i in range(first_item, end_item):
                _addLConstr(yvars.sum(i, '*'), EQ, x_j, '_f')

            # Add logical constraints between yvars and zvars
            for k in range(n_knapsacks):
                lhs = quicksum(yvars[i,k] for i in range(first_item, end_item))
                _addLConstr(lhs, LEQ, (end_item - first_item) * zvars[j,k], '_z')

            # Add logical constraints between zvars and uvars
            _addLConstr(zvars.sum(j, '*'), LEQ, uvars[j] * (n_knapsacks - 1) + 1, '_s')

        # Add maximum capacity constraints
        for k, resources in enumerate(knapsacks):
            for r, capacity in enumerate(resources):
                coeffs = {(i,k): items[i][r] for i in range(n_items)}
                _addLConstr(yvars.prod(coeffs, '*', k), LEQ, capacity, '_k')

        model.update()

    return ModelVars(model, xvars, yvars, zvars, uvars)
