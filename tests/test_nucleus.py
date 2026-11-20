import random
import unittest
from pathlib import Path

from nucleus import (
    Coordinator,
    ProposalExpired,
    ProposalNotReady,
    Shard,
    Status,
    TOLERANCE,
    reference_sample,
    uniform_u,
)
from read_capture import read_capture
from replay import load_grammar, replay

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = read_capture(ROOT / "samples" / "capture.json")


def make_coordinator(scores, parts, grammar, seed=582, order=None):
    shards = [Shard(i, ids, scores) for i, ids in enumerate(parts)]
    coord = Coordinator(shards, grammar, seed, order=order)
    coord.set_available(range(len(parts)))
    return coord


class CaptureReplayTests(unittest.TestCase):
    """按采集重放:状态区分、过期拒绝、接受推进与交换量。"""

    def test_capture_replay_outcomes(self):
        result = replay(CAPTURE)
        proposals = {r.proposal.pid: r.proposal for r in result.records if r.proposal}
        # 分片未送达 -> 无法证明核边界;配额再多也不能用部分候选当分母
        self.assertEqual(proposals["a"].status, Status.NEED_MORE)
        self.assertEqual(proposals["a"].need["reason"], "missing_shards")
        self.assertEqual(proposals["a"].need["shards"], [2])
        # 同一逻辑步:改配额(b)、调修订(c)都得到确定建议
        self.assertEqual(proposals["b"].status, Status.READY)
        self.assertEqual((proposals["b"].token, proposals["b"].nucleus_size), (4, 2))
        self.assertEqual(proposals["b"].boundary_score, 4.0)
        self.assertEqual(proposals["c"].status, Status.READY)
        self.assertEqual(proposals["c"].token, 7)
        # p=1 保留全部合法有限分数
        self.assertEqual(proposals["d"].nucleus_size, 3)
        self.assertIsNone(proposals["d"].boundary_score)
        # accept b 被拒绝:修订已被 c 改变;accept c/d 推进
        self.assertIsNotNone(result.records[5].error)  # accept b
        self.assertEqual(result.tokens, [7, 6])
        coord = result.coordinator
        self.assertEqual((coord.step, coord.state, coord.accepted_count), (2, "start", 2))

    def test_capture_tokens_match_unsharded_reference(self):
        scores = CAPTURE["scores"]
        grammar = load_grammar(CAPTURE)
        # 被接受的 c:步 0、修订 1、状态 start、已接受计数 0
        self.assertEqual(
            reference_sample(scores["0:1"], set(grammar["start"]), 1.0, 0.61, 582, 0),
            (7, 2, 4.0),
        )
        # 被接受的 d:步 1、修订 0、状态 middle、已接受计数 1
        self.assertEqual(
            reference_sample(scores["1:0"], set(grammar["middle"]), 0.8, 1.0, 582, 1),
            (6, 3, None),
        )

    def test_capture_step_exchange_volumes(self):
        result = replay(CAPTURE)
        accepts = [r.result for r in result.records if r.result]
        # 第 0 步实际交换 = a(0) + b(候选3 摘要4) + c(候选3 摘要5)
        self.assertEqual((accepts[0].stats.candidates, accepts[0].stats.summaries), (6, 9))
        self.assertEqual((accepts[1].stats.candidates, accepts[1].stats.summaries), (3, 3))

    def test_resharding_and_order_do_not_change_proposals(self):
        base = replay(CAPTURE)
        alt = replay(CAPTURE, parts_key="alternate_parts")
        shuffled = replay(CAPTURE, order=[2, 0, 1])
        for other in (alt, shuffled):
            self.assertEqual(other.tokens, base.tokens)
            for r1, r2 in zip(base.records, other.records):
                if r1.proposal and r2.proposal:
                    p1, p2 = r1.proposal, r2.proposal
                    self.assertEqual(p1.status, p2.status)
                    self.assertEqual(
                        (p1.token, p1.nucleus_size, p1.boundary_score),
                        (p2.token, p2.nucleus_size, p2.boundary_score),
                    )


class ContractTests(unittest.TestCase):
    """数值契约:均匀数、阈值容差、并列次序、p=1。"""

    def test_uniform_contract_vector(self):
        self.assertEqual(uniform_u(582, 0), 0.5793773910383468)
        self.assertEqual(uniform_u(582, 1), 0.3545439131847412)
        self.assertTrue(0.0 <= uniform_u(-1, 2**64 - 1) < 1.0)

    def test_threshold_tolerance(self):
        # 两个等质量合法词:累积到第一词的概率恰为 0.5
        scores = {"0:0": [1.0, 1.0]}
        grammar = {"start": {0: "start", 1: "start"}}
        for p, expected_size in ((0.5, 1), (0.5 + 5e-13, 1), (0.5 + 5e-11, 2)):
            coord = make_coordinator(scores, [[0, 1]], grammar)
            prop = coord.propose("x", 0, 0, 1.0, p, 50)
            self.assertEqual(prop.status, Status.READY)
            self.assertEqual(prop.nucleus_size, expected_size, f"p={p}")
            self.assertAlmostEqual(p - TOLERANCE, p - 1e-12)

    def test_ties_broken_by_global_token_id(self):
        scores = {"0:0": [3.0, 3.0, 3.0]}
        grammar = {"start": {0: "start", 1: "start", 2: "start"}}
        coord = make_coordinator(scores, [[2, 0, 1]], grammar)
        prop = coord.propose("x", 0, 0, 1.0, 1e-9, 50)
        self.assertEqual(prop.token, 0)  # 同分按全局词号升序,核只含词 0
        self.assertEqual(prop.nucleus_size, 1)

    def test_p_one_keeps_all_legal_finite(self):
        scores = {"0:0": [1.0, None, 2.0, 0.0]}
        grammar = {"start": {0: "start", 1: "start", 2: "start", 3: "start"}}
        coord = make_coordinator(scores, [[0, 1], [2, 3]], grammar)
        prop = coord.propose("x", 0, 0, 1.0, 1.0, 50)
        self.assertEqual(prop.status, Status.READY)
        self.assertEqual(prop.nucleus_size, 3)  # None(负无穷)不进入质量
        self.assertEqual(prop.legal_count, 3)

    def test_temperature_divides_scores_first(self):
        # 温度只重标定质量,不改变排序;与参照实现一致
        scores = {"0:0": [4.0, 2.0, 1.0]}
        grammar = {"start": {0: "start", 1: "start", 2: "start"}}
        for temp in (0.5, 1.0, 2.5):
            coord = make_coordinator(scores, [[0, 1, 2]], grammar)
            prop = coord.propose("x", 0, 0, temp, 0.9, 50)
            ref = reference_sample(scores["0:0"], {0, 1, 2}, temp, 0.9, 582, 0)
            self.assertEqual(prop.token, ref[0])
            self.assertEqual(prop.nucleus_size, ref[1])


class LifecycleTests(unittest.TestCase):
    """建议生命周期:接受才推进、可回查、过期不可接受、配额与补片。"""

    def setUp(self):
        self.scores = {"0:0": [0, 2, 4, 2, 4, 3, 80, None], "0:1": [0, 2, 1, 2, 4, 3, 80, 4]}
        self.grammar = {"start": {0: "middle", 2: "middle", 4: "middle", 5: "middle", 7: "middle"},
                        "middle": {1: "start", 3: "start", 6: "start"}}
        self.parts = [[4, 1, 5], [7, 0, 3], [2, 6]]

    def test_only_accept_advances(self):
        coord = make_coordinator(self.scores, self.parts, self.grammar)
        first = coord.propose("p1", 0, 0, 1.0, 0.61, 25)
        coord.discard("p1")
        self.assertEqual((coord.step, coord.state, coord.accepted_count), (0, "start", 0))
        # 放弃不消耗随机数位置:同输入再建议,词号不变
        second = coord.propose("p2", 0, 0, 1.0, 0.61, 25)
        self.assertEqual(second.token, first.token)
        with self.assertRaises(ProposalExpired):
            coord.accept("p1")  # 已放弃
        result = coord.accept("p2")
        self.assertEqual((result.step, result.previous_state, result.state), (1, "start", "middle"))
        self.assertEqual(coord.accepted_count, 1)

    def test_expired_proposal_cannot_be_accepted_but_can_be_inspected(self):
        coord = make_coordinator(self.scores, self.parts, self.grammar)
        coord.propose("old", 0, 0, 1.0, 0.61, 25)
        coord.propose("new", 0, 1, 1.0, 0.61, 25)  # 调整分数修订
        with self.assertRaises(ProposalExpired):
            coord.accept("old")
        inspected = coord.get_proposal("old")  # 过期建议仍可回查
        self.assertEqual(inspected.status, Status.READY)
        self.assertEqual(inspected.token, 4)
        coord.accept("new")
        with self.assertRaises(ProposalExpired):
            coord.accept("new")  # 步已推进

    def test_temperature_or_p_change_expires(self):
        coord = make_coordinator(self.scores, self.parts, self.grammar)
        coord.propose("a", 0, 0, 1.0, 0.61, 25)
        coord.propose("b", 0, 0, 0.8, 0.61, 25)
        with self.assertRaises(ProposalExpired):
            coord.accept("a")
        coord.accept("b")

    def test_not_ready_proposal_cannot_be_accepted(self):
        coord = make_coordinator(self.scores, self.parts, self.grammar)
        prop = coord.propose("a", 0, 0, 1.0, 0.61, 0)
        self.assertEqual(prop.status, Status.NEED_MORE)
        self.assertEqual(prop.need["reason"], "quota")
        with self.assertRaises(ProposalNotReady):
            coord.accept("a")

    def test_missing_shard_then_filled(self):
        coord = make_coordinator(self.scores, self.parts, self.grammar)
        coord.available.discard(2)
        prop = coord.propose("a", 0, 0, 1.0, 0.61, 25)
        self.assertEqual((prop.status, prop.need["reason"]), (Status.NEED_MORE, "missing_shards"))
        coord.set_available([2])  # 补缺的分片
        prop = coord.propose("b", 0, 0, 1.0, 0.61, 25)
        self.assertEqual(prop.status, Status.READY)
        self.assertEqual(prop.token, 4)

    def test_quota_segmentation_does_not_change_proposal(self):
        # 配额分段路径不同,就绪后的建议相同
        coord = make_coordinator(self.scores, self.parts, self.grammar)
        coord.propose("a", 0, 0, 1.0, 0.61, 0)
        coord.propose("b", 0, 0, 1.0, 0.61, 2)
        grown = coord.propose("c", 0, 0, 1.0, 0.61, 25)
        direct = make_coordinator(self.scores, self.parts, self.grammar).propose(
            "x", 0, 0, 1.0, 0.61, 25
        )
        self.assertEqual(grown.status, Status.READY)
        self.assertEqual(
            (grown.token, grown.nucleus_size, grown.boundary_score),
            (direct.token, direct.nucleus_size, direct.boundary_score),
        )

    def test_no_candidates_is_reported(self):
        scores = {"0:0": [None, None]}
        grammar = {"start": {0: "start", 1: "start"}}
        coord = make_coordinator(scores, [[0, 1]], grammar)
        prop = coord.propose("a", 0, 0, 1.0, 0.9, 25)
        self.assertEqual(prop.status, Status.NO_CANDIDATES)
        with self.assertRaises(ProposalNotReady):
            coord.accept("a")
        empty = make_coordinator(scores, [[0, 1]], {"start": {}})
        self.assertEqual(empty.propose("a", 0, 0, 1.0, 0.9, 25).status, Status.NO_CANDIDATES)

    def test_stale_step_and_duplicate_pid_rejected(self):
        coord = make_coordinator(self.scores, self.parts, self.grammar)
        coord.propose("a", 0, 0, 1.0, 0.61, 25)
        coord.accept("a")
        with self.assertRaises(ValueError):
            coord.propose("b", 0, 0, 1.0, 0.61, 25)  # 已推进到第 1 步
        with self.assertRaises(ValueError):
            coord.propose("a", 1, 0, 1.0, 0.61, 25)  # 标识重复

    def test_extreme_scores_keep_max_term_mass_one(self):
        # 稳定偏移使最大分项质量恒为 1:极端分数下分母也不会下溢
        scores = {"0:0": [0.0, -1e9, -2e9]}
        grammar = {"start": {0: "start", 1: "start", 2: "start"}}
        coord = make_coordinator(scores, [[0, 1], [2]], grammar)
        prop = coord.propose("a", 0, 0, 1e-6, 0.9, 50)
        self.assertEqual(prop.status, Status.READY)
        self.assertEqual((prop.token, prop.nucleus_size), (0, 1))
        ref = reference_sample(scores["0:0"], {0, 1, 2}, 1e-6, 0.9, 582, 0)
        self.assertEqual(prop.token, ref[0])


class RandomizedCrossCheckTests(unittest.TestCase):
    """随机配置下,分片协调者与未切分参照一致,且重分片不改变建议。"""

    def test_random_configs_match_reference(self):
        rng = random.Random(20261120)
        for trial in range(200):
            n = rng.randint(3, 14)
            scores = {
                "0:0": [
                    rng.choice([None, rng.uniform(-10, 10), rng.randint(-3, 3)])
                    for _ in range(n)
                ]
            }
            legal = {t for t in range(n) if rng.random() < 0.7}
            grammar = {"start": {t: "start" for t in legal}}
            temp = rng.uniform(0.3, 2.0)
            top_p = rng.choice([0.5, 0.8, 0.95, 1.0])
            seed = rng.randint(-1000, 1000)

            def random_parts():
                ids = list(range(n))
                rng.shuffle(ids)
                k = rng.randint(1, 4)
                parts = [[] for _ in range(k)]
                for i, t in enumerate(ids):
                    parts[i % k].append(t)
                return parts

            tokens = []
            for parts in (random_parts(), random_parts()):
                coord = make_coordinator(scores, parts, grammar, seed=seed)
                prop = coord.propose("x", 0, 0, temp, top_p, 100 * n)
                tokens.append((prop.status, prop.token, prop.nucleus_size))
            ref = reference_sample(scores["0:0"], legal, temp, top_p, seed, 0)
            if ref is None:
                self.assertEqual(tokens[0][0], Status.NO_CANDIDATES)
            else:
                self.assertEqual(tokens[0][0], Status.READY)
                self.assertEqual(tokens[0][1:], (ref[0], ref[1]))
            self.assertEqual(tokens[0], tokens[1], "重分片不应改变建议")


if __name__ == "__main__":
    unittest.main()
