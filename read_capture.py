import argparse
import json
from pathlib import Path


def read_capture(path):
    """读取一次实验交接采集，保留原始事件顺序。"""
    with Path(path).open(encoding="utf-8") as stream:
        capture = json.load(stream)
    events = capture.get("events") if isinstance(capture, dict) else None
    if not isinstance(events, list) or any(
        not isinstance(event, list) or not event or not isinstance(event[0], str)
        for event in events
    ):
        raise ValueError("采集资料含有无效的事件列表")
    return capture


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看交接资料的事件顺序")
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    capture = read_capture(args.capture)
    for index, event in enumerate(capture["events"], start=1):
        print(f"{index:02} {event[0]}")
