# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

result_root_candidates = [
    Path('/export2/kong102/energy_efficient_serving_results'),
    Path('~/energy_efficient_serving_results').expanduser(),
]
for c in result_root_candidates:
    if c.exists():
        RESULT_ROOT = c
        break
