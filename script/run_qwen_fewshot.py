#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module.qwen.run_tpgdiff_fewshot_qwen3vl import main

if __name__ == "__main__":
    main()
