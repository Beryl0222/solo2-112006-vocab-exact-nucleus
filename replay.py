"""生成入口:本地重放交接采集中的交互序列。

读取采集(原始数据与接收顺序),在本地进程内模拟分片节点与协调者
的交换,按事件顺序执行 available / proposal / accept,输出每条建议
的状态、接受得到的词号与可续状态,以及每一步实际交换的候选值与
摘要量,便于比较正常及阈值接近时的通信支出。

用法::

    python -m replay samples/capture.json
    python -m replay samples/capture.json --alternate   # 换用 alternate_parts 重分片
    python -m replay samples/capture.json --order 2,0,1 # 改变分片交换顺序
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from nucleus import Coordinator, ProposalError, Shard, Status
from read_capture import read_capture


@dataclass
class Record:
    """一条事件的重放结果。"""

    event: list
    line: str
    proposal: object = None
    result: object = None
    error: Exception = None


@dataclass
class ReplayResult:
    tokens: list
    records: list
    coordinator: Coordinator


def load_grammar(capture):
    return {
        state: {int(token): nxt for token, nxt in trans.items()}
        for state, trans in capture["grammar"].items()
    }


def format_proposal(event, proposal):
    _, pid, step, revision, temperature, top_p, quota = event
    head = f"proposal {pid} 步={step} 修订={revision} 温度={temperature} p={top_p} 配额={quota}"
    stats = f"交换:候选{proposal.stats.candidates} 摘要{proposal.stats.summaries}"
    if proposal.status is Status.READY:
        return (
            f"{head} -> 可抽样 词号={proposal.token} 核大小={proposal.nucleus_size} "
            f"边界={proposal.boundary_score} {stats}"
        )
    if proposal.status is Status.NO_CANDIDATES:
        return f"{head} -> 无合法候选,本步没有抽样 {stats}"
    need = proposal.need or {}
    if need.get("reason") == "missing_shards":
        detail = f"未知分片={need['shards']}"
    else:
        detail = f"配额不足 已用{need.get('units')}/预算{need.get('budget')}"
    return f"{head} -> 无法证明核边界({detail}){stats}"


def format_accept(pid, result):
    return (
        f"accept {pid} -> 词号={result.token} "
        f"状态 {result.previous_state}->{result.state} 推进到步={result.step} "
        f"本步交换:候选{result.stats.candidates} 摘要{result.stats.summaries}"
    )


def replay(capture, *, parts_key="parts", order=None):
    """按采集中的事件顺序本地重放交互,返回接受的词号与全过程记录。"""
    shards = [
        Shard(index, token_ids, capture["scores"])
        for index, token_ids in enumerate(capture[parts_key])
    ]
    coordinator = Coordinator(
        shards, load_grammar(capture), capture["seed"], order=order
    )
    records = []
    tokens = []
    for event in capture["events"]:
        kind = event[0]
        if kind == "available":
            coordinator.set_available(event[1])
            records.append(
                Record(event, f"available {event[1]} -> 已送达 {sorted(coordinator.available)}")
            )
        elif kind == "proposal":
            _, pid, step, revision, temperature, top_p, quota = event
            proposal = coordinator.propose(
                pid, step, revision, float(temperature), float(top_p), quota
            )
            records.append(Record(event, format_proposal(event, proposal), proposal=proposal))
        elif kind == "accept":
            pid = event[1]
            try:
                result = coordinator.accept(pid)
            except ProposalError as exc:
                records.append(Record(event, f"accept {pid} -> 拒绝:{exc}", error=exc))
            else:
                tokens.append(result.token)
                records.append(Record(event, format_accept(pid, result), result=result))
        else:
            raise ValueError(f"未知事件 {kind!r}")
    return ReplayResult(tokens, records, coordinator)


def main(argv=None):
    parser = argparse.ArgumentParser(description="本地重放交接采集中的交互")
    parser.add_argument("capture", type=Path)
    parser.add_argument(
        "--alternate", action="store_true", help="使用 alternate_parts 分片方案"
    )
    parser.add_argument("--order", type=str, default=None, help="分片交换顺序,如 2,0,1")
    args = parser.parse_args(argv)
    capture = read_capture(args.capture)
    parts_key = "alternate_parts" if args.alternate else "parts"
    order = [int(item) for item in args.order.split(",")] if args.order else None
    result = replay(capture, parts_key=parts_key, order=order)
    for record in result.records:
        print(record.line)
    print(f"接受词号序列: {result.tokens}")
    candidates = sum(r.result.stats.candidates for r in result.records if r.result)
    summaries = sum(r.result.stats.summaries for r in result.records if r.result)
    print(f"接受步合计交换: 候选{candidates} 摘要{summaries}")


if __name__ == "__main__":
    main()
