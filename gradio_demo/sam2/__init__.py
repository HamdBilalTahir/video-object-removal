# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from hydra import initialize

from .build_sam import load_model

# Workaround for Gradio hot-reloader causing KeyError: '__main__' in hydra
if '__main__' not in sys.modules:
    import builtins
    sys.modules['__main__'] = builtins

initialize("configs", version_base="1.2")
