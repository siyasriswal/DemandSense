"""Tests run against a throw-away copy of the data so they never change the real project files."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="demandsense_test_"))
(TMP / "data" / "raw").mkdir(parents=True)
shutil.copyfile(ROOT / "data" / "raw" / "demand_dataset_v2_items.csv", TMP / "data" / "raw" / "demand_dataset_v2_items.csv")
(TMP / "reports").mkdir()
shutil.copyfile(ROOT / "reports" / "phase2_benchmark.csv", TMP / "reports" / "phase2_benchmark.csv")
os.environ["DEMANDSENSE_HOME"] = str(TMP)
sys.path.insert(0, str(ROOT))
