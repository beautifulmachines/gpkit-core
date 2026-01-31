"IR dispatch functions for reconstructing gpkit objects from IR dicts."


def constraint_from_ir(ir_dict, var_registry):
    """Reconstruct a constraint from its IR dict.

    Parameters
    ----------
    ir_dict : dict
        IR dict with "type", "oper", "left", "right", and optional "lineage".
    var_registry : dict
        Mapping from var_ref strings to VarKey objects.

    Returns
    -------
    constraint : ScalarSingleEquationConstraint
    """
    from .nomials.math import (
        MonomialEquality,
        PosynomialInequality,
        SignomialInequality,
        SingleSignomialEquality,
    )

    type_map = {
        "PosynomialInequality": PosynomialInequality,
        "MonomialEquality": MonomialEquality,
        "SignomialInequality": SignomialInequality,
        "SingleSignomialEquality": SingleSignomialEquality,
    }

    constraint_type = ir_dict["type"]
    if constraint_type not in type_map:
        raise ValueError(f"Unknown constraint type: {constraint_type}")
    cls = type_map[constraint_type]
    return cls.from_ir(ir_dict, var_registry)
