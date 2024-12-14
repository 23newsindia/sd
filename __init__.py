import os
import sys
import importlib.util
import folder_paths

# Add the current directory to Python path
STABLEANIMATOR_DIR = os.path.dirname(os.path.realpath(__file__))
if STABLEANIMATOR_DIR not in sys.path:
    sys.path.append(STABLEANIMATOR_DIR)

# Add model folder path
folder_paths.add_model_folder_path("stableanimator", os.path.join(STABLEANIMATOR_DIR, "models"))

# Import nodes.py
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

nodes_path = os.path.join(os.path.dirname(__file__), "nodes.py")
if os.path.exists(nodes_path):
    import nodes
    NODE_CLASS_MAPPINGS.update(nodes.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(nodes.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']