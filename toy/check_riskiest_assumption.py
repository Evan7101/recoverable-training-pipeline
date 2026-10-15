"""风险最高假设的最小检查：JSON checkpoint 往返 + 分段恢复是否位级等价于不间断运行。"""
import json, pathlib

CKPT_PATH = pathlib.Path(__file__).resolve().parent / "checkpoint.json"

def train_segment(state, steps):
    for _ in range(steps):
        state['weight'] -= 0.1 * (2 * (state['weight'] - 3))
        state['step'] += 1
    return state

# A：不间断运行 10 步（基线语义）
uninterrupted = train_segment({'step': 0, 'weight': 0.0}, 10)

# B：中断/恢复，4 步 -> 写入 checkpoint.json -> 读回 -> 再 6 步
resumed = train_segment({'step': 0, 'weight': 0.0}, 4)
CKPT_PATH.write_text(json.dumps(resumed))
resumed = json.loads(CKPT_PATH.read_text())
resumed = train_segment(resumed, 6)

bit_exact = (uninterrupted['step'] == resumed['step']
             and uninterrupted['weight'] == resumed['weight']
             and repr(uninterrupted['weight']) == repr(resumed['weight']))

print(f"uninterrupted: step={uninterrupted['step']} weight={uninterrupted['weight']!r}")
print(f"resumed      : step={resumed['step']} weight={resumed['weight']!r}")
print(f"bit-exact equal: {bit_exact}")
print(f"checkpoint keys saved: {sorted(json.loads(CKPT_PATH.read_text()).keys())}")
