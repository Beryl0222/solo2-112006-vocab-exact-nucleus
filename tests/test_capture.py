from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from read_capture import read_capture


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "samples" / "capture.json"


class CaptureTests(unittest.TestCase):
    def test_event_order_and_dependency(self):
        self.assertEqual(np.__version__, "2.2.6")
        events = read_capture(CAPTURE)["events"]
        self.assertEqual([event[0] for event in events], ['available', 'proposal', 'available', 'proposal', 'proposal', 'accept', 'accept', 'proposal', 'accept'])

    def test_invalid_event_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"events":[[]]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "事件列表"):
                read_capture(path)

    def test_command_reads_the_file(self):
        result = subprocess.run(
            [sys.executable, "-m", "read_capture", str(CAPTURE)], cwd=ROOT,
            text=True, capture_output=True, check=True,
        )
        self.assertEqual(len(result.stdout.splitlines()), 9)

    def test_capture_geometry_and_values(self):
        capture = read_capture(CAPTURE)
        self.assertEqual(sorted(w for shard in capture["parts"] for w in shard), list(range(8)))
        self.assertEqual(sorted(w for shard in capture["alternate_parts"] for w in shard), list(range(8)))
        self.assertEqual(capture["scores"]["0:0"][2], capture["scores"]["0:0"][4])
