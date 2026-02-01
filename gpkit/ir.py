"IR dispatch functions for reconstructing gpkit objects from IR dicts."

import json
from pathlib import Path


def to_json(model, path=None):
    """Serialize a Model to JSON.

    Parameters
    ----------
    model : gpkit.Model
        The model to serialize.
    path : str or Path, optional
        If given, write JSON to this file path. Otherwise return JSON string.

    Returns
    -------
    str or None
        JSON string if path is None, otherwise None (written to file).
    """
    ir = model.to_ir()
    json_str = json.dumps(ir, indent=2)
    if path is not None:
        Path(path).write_text(json_str)
        return None
    return json_str


def from_json(json_str_or_path):
    """Deserialize a Model from JSON.

    Parameters
    ----------
    json_str_or_path : str or Path
        Either a JSON string or a path to a JSON file.

    Returns
    -------
    gpkit.Model
        A solvable Model reconstructed from the IR.
    """
    from .model import Model

    if isinstance(json_str_or_path, Path):
        ir_doc = json.loads(json_str_or_path.read_text())
    else:
        path = Path(json_str_or_path)
        try:
            if path.exists():
                ir_doc = json.loads(path.read_text())
            else:
                ir_doc = json.loads(json_str_or_path)
        except OSError:
            # Path too long or invalid — treat as JSON string
            ir_doc = json.loads(json_str_or_path)
    return Model.from_ir(ir_doc)


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
