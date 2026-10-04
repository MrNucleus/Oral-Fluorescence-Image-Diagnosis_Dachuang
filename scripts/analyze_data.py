"""数据探索入口：python scripts/analyze_data.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.analyze import analyze


if __name__ == "__main__":
    analyze()
