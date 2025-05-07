# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

if 'IS_GOOGLE_CLOUD' in os.environ:
    RESULT_ROOT = Path('~/energy_efficient_serving_results').expanduser()
else:
    RESULT_ROOT = Path('/export2/kong102/energy_efficient_serving_results')
