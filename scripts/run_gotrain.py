"""Run the GoTrain agent without installing the package. Same arguments as `gotrain`:

    python scripts/run_gotrain.py run --task m-eurosat --budget-min 30 --phase tune

Overnight on SageMaker (no tmux there), one task at a time:

    nohup python scripts/run_gotrain.py run --task m-eurosat --budget-min 120 \
        > ~/gotrain_home/nohup-m-eurosat.out 2>&1 &
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gotrain.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
