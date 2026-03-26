"""Inject a lightweight edgar stub before any test imports pitedgar.mapping.

edgartools >=5 performs a network download at import time, which hangs
test collection in environments without network access or when the SEC
endpoint is slow. By placing a stub in sys.modules first, the real
edgar package is never imported during the test suite.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock

# Only install the stub if edgar hasn't already been imported (e.g. in
# an integration test environment where the real library is needed).
if "edgar" not in sys.modules:
    _edgar_stub = ModuleType("edgar")
    _edgar_stub.set_identity = MagicMock()
    _edgar_stub.Company = MagicMock()
    sys.modules["edgar"] = _edgar_stub
