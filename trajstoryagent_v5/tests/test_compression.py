import itertools
import math

import pytest

from trajstoryagent_v5.v5_framework.conditional_math import aggregate
from trajstoryagent_v5 import TrajStoryAgent
from trajstoryagent_v5.v5_framework.compression_math import compress_with_exceptions
from trajstoryagent_v5.v5_framework import refinement


def test_name_only_cost_matches_exhaustive_search():
    leaves = [dict(n=n, mean=y, sse=n, sd=1., condition={'g': str(i)})
              for i, (n, y) in enumerate([(500, 0.), (500, .2), (30, .4)])]
    total = aggregate(leaves)
    p = dict(leaves=leaves, total=total, scale=total['sse']/total['n'])
    result = compress_with_exceptions(p)
    costs = []
    for base in {total['mean'], *[l['mean'] for l in leaves]}:
        for flags in itertools.product([False, True], repeat=3):
            reconstructed = [l['mean'] if flag else base for l, flag in zip(leaves, flags)]
            errors = [(l['mean']-v)**2/p['scale'] for l,v in zip(leaves, reconstructed)]
            if any(e > .25**2+1e-12 for e in errors):
                continue
            if any((l['mean']-total['mean'])*(v-total['mean']) < -1e-12
                   for l,v in zip(leaves, reconstructed)):
                continue
            costs.append(math.log2(3)*(1+sum(flags)) + sum(l['n']*e for l,e in zip(leaves,errors)))
    assert result['objective'] == pytest.approx(min(costs))
    assert result['code_length'] == pytest.approx(math.log2(3)*(1+len(result['exception_indices'])))
    assert result['max_error_in_root_sd'] <= .25


def test_default_agent_and_branch_use_new_kernel(tmp_path, monkeypatch):
    agent = TrajStoryAgent(None, generator='gpt')
    assert agent.refiner.func is refinement.run_branch
    assert agent.refiner.keywords == {"min_support": 1}
    measure = dict(description='duration', unit='minutes', observation_unit='trip', scope='all',
                   conditions=['g'], cells=[dict(condition={'g':g}, n=100, mean=y, sse=1.)
                                           for g,y in [('a', 0.), ('b', 10.)]])
    monkeypatch.setattr(refinement, 'observations', lambda *args: {'measures': [measure]})
    def interpret(slug, evidence, bundle, facts, folder):
        refinement.write(folder/'interpretation.json', {'use_refinement': True, 'insight': 'refined'})
        return 'refined'
    monkeypatch.setattr(refinement, 'interpret', interpret)
    evidence = {'question_id':'Q1', 'question':'Compare?', 'insight':'original'}
    branch = agent.refiner('gpt', evidence, {}, {}, tmp_path/'branch')
    cert = refinement.read(tmp_path/'branch/mathematical_certificate.json')
    compression = cert['compressions'][0]
    assert cert['compression_cost_model'] == refinement.COST_MODEL
    assert compression['code_length'] == 1 + len(compression['exception_indices'])
    assert branch['insights']['conditional_compressed'] == 'refined'
    # A branch produced under the old policy cannot masquerade as new output.
    branch.pop('compression_cost_model')
    refinement.write(tmp_path/'branch/branch.json', branch)
    with pytest.raises(ValueError, match='different compression cost'):
        agent.refiner('gpt', evidence, {}, {}, tmp_path/'branch')
