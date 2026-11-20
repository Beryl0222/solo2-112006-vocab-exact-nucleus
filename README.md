# 分片词表的精确核采样与续跑

`samples/capture.json` 是离线合成的交接采集，包含原始数据和接收顺序。[交接协议](交接协议.md) 描述各字段及上下游共同使用的数值口径。当前入口可读取采集文件并列出操作顺序。

`nucleus.py` 实现协议两侧:分片算子 `Shard`(摘要、有序边界探针、分段候选)与协调者 `Coordinator`(建议-接受生命周期、配额与交换量统计),`reference_sample` 给出未切分词表的参照抽样用于对照。`replay.py` 是生成入口,在本地重放采集中的交互序列,输出每条建议的状态、接受得到的词号与可续状态,以及每一步实际交换的候选值与摘要量。

环境使用 Python 3.11 和锁定的 NumPy。首次在本地准备虚拟环境后运行以下命令。

```bash
python -m pip install -r requirements.txt
python -m read_capture samples/capture.json
python -m replay samples/capture.json               # 本地重放交互
python -m replay samples/capture.json --alternate   # 换用 alternate_parts 重分片,建议不变
python -m replay samples/capture.json --order 2,0,1 # 改变分片交换顺序,建议不变
python -m unittest discover -s tests -v
```
