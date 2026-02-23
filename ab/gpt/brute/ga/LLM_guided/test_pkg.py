import sys
import pkg_resources
from pathlib import Path

try:
    dist = pkg_resources.get_distribution('nn-dataset')
    print(f"Location: {dist.location}")
except Exception as e:
    print(f"Error: {e}")
