"""分片词表上的核采样:分片算子、协调者与建议生命周期。

词表分数分散在多个分片上,协调者不把每步全量分数拉回同一进程,
只通过协议允许的三种交换完成与未切分词表一致的 top-p 抽样:

* ``Shard.summary``     固定长度摘要:合规有限项数、最大分数、稳定质量;
* ``Shard.probe``       给定有序边界(分数, 词号)之前的合规计数与质量;
* ``Shard.candidates``  按偏移分段的顶部候选 ``[词号, 分数]``。

数值口径与交接协议一致(见 交接协议.md):

* 分数先除正温度;偏移不改变质量,摘要与探针以本片最大分数为稳定偏移。
* 候选按归一化前分数降序,同分按全局词号升序。
* 核包含累计质量第一次达到 p 的词;FP64 比较,距离概率阈值不超过
  ``TOLERANCE``(1e-12)视为相等。p 为 1 时保留所有合法有限分数,
  概率在核内重归一化。
* 均匀数由 BLAKE2b(digest_size=8) 作用于种子(有符号 64 位大端)与
  已接受计数(无符号 64 位大端)的拼接,取摘要大端整数的高 53 位乘
  2**-53;核内按排序累积概率,第一个严格超过它的词被选中。
* 一个配额单位是一条候选或一次固定长度摘要交换;``quota`` 是当次
  额外允许的单位数,基线是每个已送达分片各一次摘要。

建议有效期由步、状态、分数修订、温度、p 及种子共同决定;只有接受
才推进词法状态与已接受计数(实际抽样位置)。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum

TOLERANCE = 1e-12


def uniform_u(seed: int, accepted_count: int) -> float:
    """契约均匀数,只消耗已接受的随机数位置。"""
    data = seed.to_bytes(8, "big", signed=True) + accepted_count.to_bytes(8, "big")
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return (int.from_bytes(digest, "big") >> 11) * 2**-53


def scores_key(step: int, revision: int) -> str:
    return f"{step}:{revision}"


@dataclass(frozen=True)
class Summary:
    """固定长度摘要:合规有限项数、最大分数、以本片最大分数为偏移的稳定质量。"""

    count: int
    max_score: float | None
    mass: float


@dataclass(frozen=True)
class Probe:
    """有序边界之前的合规项计数与质量(与摘要同一偏移口径)。"""

    count: int
    mass: float


class Shard:
    """分片算子。

    持有词表一段的原始分数数组,只暴露协议允许的可重复读取;
    协调者不得读取原始数组。``scores`` 以 ``"步:修订"`` 索引,
    按全局词号给出 logits,``None`` 代表负无穷。
    """

    def __init__(self, shard_id, token_ids, scores):
        self.shard_id = shard_id
        self.token_ids = tuple(token_ids)
        self._scores = scores

    def _legal_sorted(self, step, revision, legal):
        raw = self._scores[scores_key(step, revision)]
        items = []
        for token in self.token_ids:
            if token in legal:
                score = raw[token]
                if score is not None and math.isfinite(score):
                    items.append((token, float(score)))
        items.sort(key=lambda item: (-item[1], item[0]))
        return items

    def summary(self, step, revision, legal, temperature):
        items = self._legal_sorted(step, revision, legal)
        if not items:
            return Summary(0, None, 0.0)
        shift = items[0][1] / temperature
        mass = sum(math.exp(score / temperature - shift) for _, score in items)
        return Summary(len(items), items[0][1], mass)

    def probe(self, step, revision, legal, temperature, boundary):
        """统计排序严格位于 ``(边界分数, 边界词号)`` 之前的合规项。"""
        bound_score, bound_token = boundary
        items = self._legal_sorted(step, revision, legal)
        if not items:
            return Probe(0, 0.0)
        shift = items[0][1] / temperature
        count = 0
        mass = 0.0
        for token, score in items:
            if score > bound_score or (score == bound_score and token < bound_token):
                count += 1
                mass += math.exp(score / temperature - shift)
        return Probe(count, mass)

    def candidates(self, step, revision, legal, offset, limit):
        """返回有序候选中 ``[offset, offset + limit)`` 一段的 ``[词号, 分数]``。"""
        return self._legal_sorted(step, revision, legal)[offset : offset + limit]


class Status(Enum):
    """建议状态,调用方据此分清三种情形。"""

    READY = "ready"  # 核边界已证明,可以抽样
    NO_CANDIDATES = "no_candidates"  # 真正没有合法候选,本步没有抽样
    NEED_MORE = "need_more"  # 仍无法证明核边界(分片未知或配额不足),等待进一步读取


@dataclass
class ExchangeStats:
    """一次交换量:摘要次数(含探针)与候选条数,合计即配额单位。"""

    summaries: int = 0
    candidates: int = 0

    @property
    def units(self):
        return self.summaries + self.candidates

    def merge(self, other):
        self.summaries += other.summaries
        self.candidates += other.candidates


@dataclass
class Proposal:
    """一次建议。有效期由 (步, 状态, 修订, 温度, p, 种子) 共同决定。"""

    pid: str
    step: int
    revision: int
    temperature: float
    top_p: float
    seed: int
    state: str
    status: Status
    token: int | None = None
    nucleus_size: int | None = None
    boundary_score: float | None = None
    denominator: float | None = None
    legal_count: int | None = None
    need: dict | None = None
    stats: ExchangeStats = field(default_factory=ExchangeStats)
    discarded: bool = False

    @property
    def fingerprint(self):
        return (
            self.step,
            self.state,
            self.revision,
            self.temperature,
            self.top_p,
            self.seed,
        )


class ProposalError(Exception):
    """建议生命周期错误。"""


class UnknownProposal(ProposalError):
    """建议标识不存在。"""


class ProposalNotReady(ProposalError):
    """建议尚不可接受(无合法候选或仍无法证明核边界)。"""


class ProposalExpired(ProposalError):
    """建议已过期:词法状态、步、分数修订、温度或 p 已经变化,或已被放弃。"""


@dataclass
class AcceptResult:
    """接受结果:词号、可续状态与本步实际交换量。"""

    token: int
    step: int
    state: str
    previous_state: str
    accepted_count: int
    stats: ExchangeStats


class Coordinator:
    """协调者:给出温度、p、种子与当前词法状态,驱动建议-接受流程。"""

    def __init__(self, shards, grammar, seed, start_state=None, order=None):
        self.shards = list(shards)
        self.grammar = {state: dict(trans) for state, trans in grammar.items()}
        self.seed = seed
        if start_state is not None:
            self.state = start_state
        elif "start" in self.grammar:
            self.state = "start"
        else:
            self.state = next(iter(self.grammar))
        self.step = 0
        self.accepted_count = 0
        self.order = list(order) if order is not None else list(range(len(self.shards)))
        if sorted(self.order) != list(range(len(self.shards))):
            raise ValueError("交换顺序必须是分片下标的一个排列")
        self.available = set()
        self.current_inputs = None
        self.proposals = {}
        self.step_stats = ExchangeStats()

    # ------------------------------------------------------------------
    # 调用方控制面

    def set_available(self, shard_ids):
        """登记当次已送达的分片,未送达的分片留作未知。"""
        self.available.update(shard_ids)

    def get_proposal(self, pid):
        """回查已发出的建议(包括已过期或已放弃的)。"""
        try:
            return self.proposals[pid]
        except KeyError:
            raise UnknownProposal(f"建议 {pid!r} 不存在") from None

    def discard(self, pid):
        """放弃一份建议;不推进任何状态。"""
        self.get_proposal(pid).discarded = True

    # ------------------------------------------------------------------
    # 建议

    def propose(self, pid, step, revision, temperature, top_p, quota):
        """就当前步发出建议。可多次继续同一步读取:改配额、补缺分片或调整修订。"""
        if pid in self.proposals:
            raise ValueError(f"建议标识 {pid!r} 已存在")
        if step != self.step:
            raise ValueError(f"建议针对第 {step} 步,但当前已推进到第 {self.step} 步")
        temperature = float(temperature)
        top_p = float(top_p)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("温度必须是正有限数")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("p 必须落在 (0, 1]")
        if quota < 0:
            raise ValueError("配额不能为负")
        # 最近一次建议声明的输入即刻生效,指纹不符的旧建议随之过期。
        self.current_inputs = (revision, temperature, top_p)

        proposal = Proposal(
            pid=pid,
            step=step,
            revision=revision,
            temperature=temperature,
            top_p=top_p,
            seed=self.seed,
            state=self.state,
            status=Status.NEED_MORE,
        )
        stats = ExchangeStats()
        legal = set(self.grammar.get(self.state, {}))

        missing = [i for i in self.order if i not in self.available]
        if missing:
            proposal.need = {"reason": "missing_shards", "shards": missing}
            return self._commit(proposal, stats)

        budget = len(self.shards) + quota
        summaries = {}
        for i in self.order:
            summaries[i] = self.shards[i].summary(step, revision, legal, temperature)
            stats.summaries += 1
        total = sum(item.count for item in summaries.values())
        proposal.legal_count = total
        if total == 0:
            proposal.status = Status.NO_CANDIDATES
            return self._commit(proposal, stats)

        top = max(item.max_score for item in summaries.values() if item.count)
        shift = top / temperature
        total_mass = 0.0
        for i in self.order:
            item = summaries[i]
            if item.count:
                total_mass += math.exp(item.max_score / temperature - shift) * item.mass

        # 稳定偏移保证最大分项质量恒为 1,总分母不会下溢。
        u = uniform_u(self.seed, self.accepted_count)
        limits = {i: 0 for i in self.order}
        fetched = {i: [] for i in self.order}

        def need_more_quota():
            proposal.need = {
                "reason": "quota",
                "units": stats.units,
                "budget": budget,
                "limits": dict(limits),
                "counts": {i: summaries[i].count for i in self.order},
            }
            return self._commit(proposal, stats)

        def grow(plan):
            cost = sum(plan[i] - limits[i] for i in plan)
            if stats.units + cost > budget:
                return False
            for i, new_limit in plan.items():
                segment = self.shards[i].candidates(
                    step, revision, legal, limits[i], new_limit - limits[i]
                )
                stats.candidates += len(segment)
                fetched[i].extend(segment)
                limits[i] = new_limit
            return True

        while True:
            merged = []
            for i in self.order:
                merged.extend((token, score, i) for token, score in fetched[i])
            merged.sort(key=lambda item: (-item[1], item[0]))

            pos = None
            cum = 0.0
            if top_p >= 1.0:
                # p 为 1:核保留全部合法有限词,直接定位累积首次严格超过 u 的词。
                for index, (_, score, _) in enumerate(merged):
                    cum += math.exp(score / temperature - shift)
                    if cum / total_mass > u:
                        pos = index
                        break
            else:
                for index, (_, score, _) in enumerate(merged):
                    cum += math.exp(score / temperature - shift)
                    if cum / total_mass >= top_p - TOLERANCE:
                        pos = index
                        break

            if pos is None:
                growable = [i for i in self.order if limits[i] < summaries[i].count]
                if not growable:
                    if not merged:
                        return need_more_quota()  # 不可达,防御
                    pos = len(merged) - 1  # FP 边界下未能严格超过,取末位
                else:
                    plan = {
                        i: min(summaries[i].count, max(1, limits[i] * 2))
                        for i in growable
                    }
                    if not grow(plan):
                        return need_more_quota()
                    continue

            # 证明 pos 之前没有遗漏:逐片核对有序边界之前的计数。
            boundary = (merged[pos][1], merged[pos][0])
            need_probe = [i for i in self.order if limits[i] < summaries[i].count]
            if stats.units + len(need_probe) > budget:
                return need_more_quota()
            actual_before = {}
            for i in need_probe:
                probe = self.shards[i].probe(step, revision, legal, temperature, boundary)
                stats.summaries += 1
                actual_before[i] = probe.count
            incomplete = {}
            for i in self.order:
                fetched_before = sum(
                    1
                    for token, score in fetched[i]
                    if score > boundary[0]
                    or (score == boundary[0] and token < boundary[1])
                )
                if fetched_before != actual_before.get(i, fetched_before):
                    incomplete[i] = actual_before[i]
            if not incomplete:
                break
            plan = {}
            for i, actual in incomplete.items():
                extra = 1 if merged[pos][2] == i else 0
                plan[i] = actual + extra
            if not grow(plan):
                return need_more_quota()

        nucleus = merged[: pos + 1]
        if top_p >= 1.0:
            proposal.token = merged[pos][0]
            proposal.nucleus_size = total
            proposal.boundary_score = None
            proposal.denominator = total_mass
        else:
            z_nuc = cum
            chosen = nucleus[-1][0]
            running = 0.0
            for token, score, _ in nucleus:
                running += math.exp(score / temperature - shift)
                if running / z_nuc > u:
                    chosen = token
                    break
            proposal.token = chosen
            proposal.nucleus_size = len(nucleus)
            proposal.boundary_score = merged[pos][1]
            proposal.denominator = z_nuc
        proposal.status = Status.READY
        return self._commit(proposal, stats)

    def _commit(self, proposal, stats):
        proposal.stats = stats
        self.proposals[proposal.pid] = proposal
        self.step_stats.merge(stats)
        return proposal

    # ------------------------------------------------------------------
    # 接受

    def accept(self, pid):
        """接受指定建议;只有接受才推进词法状态与已接受计数。"""
        proposal = self.get_proposal(pid)
        if proposal.status is not Status.READY:
            raise ProposalNotReady(f"建议 {pid!r} 状态为 {proposal.status.value},不能接受")
        if proposal.discarded:
            raise ProposalExpired(f"建议 {pid!r} 已被放弃")
        if self.current_inputs is None or proposal.fingerprint != self._fingerprint():
            raise ProposalExpired(
                f"建议 {pid!r} 已过期:词法状态、步、分数修订、温度或 p 已经变化"
            )
        token = proposal.token
        previous = self.state
        self.state = self.grammar[self.state][token]
        self.step += 1
        self.accepted_count += 1
        stats, self.step_stats = self.step_stats, ExchangeStats()
        return AcceptResult(
            token=token,
            step=self.step,
            state=self.state,
            previous_state=previous,
            accepted_count=self.accepted_count,
            stats=stats,
        )

    def _fingerprint(self):
        revision, temperature, top_p = self.current_inputs
        return (self.step, self.state, revision, temperature, top_p, self.seed)


def reference_sample(scores, legal, temperature, top_p, seed, accepted_count):
    """未切分词表的参照抽样:同一数值契约,直接对全词表计算。

    返回 ``(词号, 核大小, 边界分数)``;没有合法有限词时返回 ``None``。
    用于对照验证分片协调者的建议与未切分结果一致。
    """
    items = [
        (token, float(score))
        for token, score in enumerate(scores)
        if token in legal and score is not None and math.isfinite(score)
    ]
    if not items:
        return None
    items.sort(key=lambda item: (-item[1], item[0]))
    shift = items[0][1] / temperature
    masses = [math.exp(score / temperature - shift) for _, score in items]
    total = 0.0
    for mass in masses:
        total += mass
    u = uniform_u(seed, accepted_count)
    if top_p >= 1.0:
        cum = 0.0
        for (token, _), mass in zip(items, masses):
            cum += mass
            if cum / total > u:
                return token, len(items), None
        return items[-1][0], len(items), None
    cum = 0.0
    boundary = len(items) - 1
    for index, mass in enumerate(masses):
        cum += mass
        if cum / total >= top_p - TOLERANCE:
            boundary = index
            break
    z_nuc = 0.0
    for mass in masses[: boundary + 1]:
        z_nuc += mass
    running = 0.0
    for (token, _), mass in zip(items[: boundary + 1], masses[: boundary + 1]):
        running += mass
        if running / z_nuc > u:
            return token, boundary + 1, items[boundary][1]
    return items[boundary][0], boundary + 1, items[boundary][1]
