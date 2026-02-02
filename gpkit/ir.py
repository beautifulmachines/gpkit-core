"IR dispatch functions for reconstructing gpkit objects from IR dicts."

import json
from pathlib import Path

from .model import Model


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
        Path(path).write_text(json_str, encoding="utf-8")
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
    if isinstance(json_str_or_path, Path):
        ir_doc = json.loads(json_str_or_path.read_text(encoding="utf-8"))
    else:
        path = Path(json_str_or_path)
        try:
            if path.exists():
                ir_doc = json.loads(path.read_text(encoding="utf-8"))
            else:
                ir_doc = json.loads(json_str_or_path)
        except OSError:
            # Path too long or invalid — treat as JSON string
            ir_doc = json.loads(json_str_or_path)
    return Model.from_ir(ir_doc)
