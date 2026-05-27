import pandas as pd
import numpy as np
import json
import os

def default_serializer(obj):
    """Helper to convert NumPy arrays and other objects to JSON-serializable types"""
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()  # convert NumPy arrays to Python lists
    if isinstance(obj, (np.integer,)):
        return int(obj)      # NumPy ints → Python int
    if isinstance(obj, (np.floating,)):
        return float(obj)    # NumPy floats → Python float
    return str(obj)          # fallback: stringify

for fn in os.listdir():
    break
    if not fn.endswith(".pkl"):
        continue

    name = fn.rsplit(".", 1)[0]
    print(f"📂 Processing: {fn}")

    obj = pd.read_pickle(fn)

    # Convert DataFrame directly to JSON
    if isinstance(obj, pd.DataFrame):
        obj.to_json(f"{name}.json", orient="records", indent=2, default_handler=default_serializer)

    # Convert dict / list / array into JSON
    else:
        with open(f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=default_serializer, ensure_ascii=False)

import ijson
from decimal import Decimal
import os
import json

folder = "."

def convert_decimal(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_decimal(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_decimal(v) for k, v in obj.items()}
    else:
        return obj

for fn in os.listdir(folder):
    if not fn.endswith(".json"):
        continue

    file_path = os.path.join(folder, fn)
    print(f"\n📂 File: {fn}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            sample = []
            for i, item in enumerate(ijson.items(f, "item")):
                sample.append(convert_decimal(item))
                if i >= 9:
                    break
        print(json.dumps(sample, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"⚠️ Error processing {fn}: {e}")
