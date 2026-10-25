"""Original prompt and metric functions, independent of experiment runners."""
import json,hashlib,statistics
from pathlib import Path
import numpy as np

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def metrics(matrix):
    maxima = list(map(max, matrix))
    return dict(g_eval=statistics.mean(maxima), coverage_at_0_6=statistics.mean((x >= 0.6 for x in maxima)), maxima=maxima)

def difference(a, b):
    delta = [y - x for x, y in zip(a, b)]
    return dict(n=len(a), mean_absolute_delta=statistics.mean((abs(d) for d in delta)), max_absolute_delta=max(map(abs, delta)), changed=sum((abs(d) > 1e-08 for d in delta)), changed_at_least_0_1=sum((abs(d) >= 0.1 - 1e-08 for d in delta)), changed_at_least_0_2=sum((abs(d) >= 0.2 - 1e-08 for d in delta)), threshold_flips=sum(((x >= 0.6) != (y >= 0.6) for x, y in zip(a, b))))

def paired(rows):
    a = np.asarray(rows)
    sample = np.random.default_rng(20261011).integers(0, len(a), (10000, len(a)))
    return dict(mean_delta=float(a.mean()), case_bootstrap_ci95=np.quantile(a[sample].mean(axis=1), [0.025, 0.975]).tolist(), wins=int((a > 1e-09).sum()), ties=int((abs(a) <= 1e-09).sum()), losses=int((a < -1e-09).sum()))

def original_prompts():
    v=json.loads((Path(__file__).parent/'judge_prompts.json').read_text(encoding='utf-8'))
    return v['G_EVAL_BASIC_TEMPLATE'],v['G_EVAL_BASIC_SYSTEM_MESSAGE']
