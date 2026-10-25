"""Frozen-observation sensitivity adapter for all four mathematical parameters."""
import traceback
from pathlib import Path
from trajstoryagent_v5.ablation import ablation_replay as replay
from trajstoryagent_v5.v5_framework import export_support as refinement
from trajstoryagent_v5.v5_framework.export_support import read, write
from trajstoryagent_v5.v5_framework.controller import TrajStoryAgent
from trajstoryagent_v5.v5_framework.conditional_math import conditional_partition, render_facts
from trajstoryagent_v5.v5_framework.compression_math import compress_with_exceptions
from trajstoryagent_v5.v5_framework.refinement import COST_MODEL
from trajstoryagent_v5.v5_framework.interpretation import interpret

def branch(slug, evidence, bundle, profile, folder, *, frozen=None, partition_penalty=1., min_support=1, tolerance_sd=.25, distortion_weight=1.):
    """Freeze metric binding for Q branches; use the identical export policy for new F branches."""
    if (folder/'branch.json').exists():
        cached = read(folder/'branch.json')
        assert all(cached[k] == v for k,v in dict(partition_penalty=partition_penalty,min_support=min_support,tolerance_sd=tolerance_sd,distortion_weight=distortion_weight).items())
        return cached
    folder.mkdir(parents=True, exist_ok=True)
    write(folder/'original_evidence.json', evidence)
    result = dict(question_id=evidence['question_id'], question=evidence['question'],
                  original_insight=evidence['insight'], status='pending',
                  compression_cost_model=COST_MODEL, schema_version=2, min_support=min_support,
                  partition_penalty=partition_penalty, tolerance_sd=tolerance_sd, distortion_weight=distortion_weight,
                  insights={a:evidence['insight'] for a in refinement.ARMS},
                  arm_status={a:'fallback' for a in refinement.ARMS})
    try:
        if frozen is not None:
            result['frozen_observations_source'] = str(frozen/'observations.json')
            if not (frozen/'observations.json').exists():
                previous = read(frozen/'branch.json')
                result.update(status=previous['status'], reason='Frozen source exported no observations; no rebinding permitted')
                write(folder/'branch.json', result)
                return result
            obs = read(frozen/'observations.json')
            write(folder/'observations.json', obs)
        else:
            obs = refinement.observations(slug, evidence, bundle, profile, folder)
        if not obs['measures']:
            result.update(status='unsupported', reason=obs.get('unsupported_reason'))
        else:
            partitions = [conditional_partition(m, penalty=partition_penalty, min_support=min_support) for m in obs['measures']]
            compressions = [compress_with_exceptions(p, distortion_weight=distortion_weight, tolerance_sd=tolerance_sd) for p in partitions]
            assert all(p['penalty']==partition_penalty and p['min_support']==min_support for p in partitions)
            write(folder/'mathematical_certificate.json', dict(partitions=partitions, compressions=compressions,
                  compression_cost_model=COST_MODEL, partition_penalty=partition_penalty))
            for a in refinement.ARMS:
                target = folder/a
                facts = [f for p,c in zip(partitions,compressions) for f in render_facts(p,c if a=='conditional_compressed' else None)]
                write(target/'facts.json', facts)
                try:
                    first = folder/'conditional_only'
                    if a=='conditional_compressed' and (first/'interpretation.json').exists() and read(first/'facts.json')==facts:
                        reused = dict(read(first/'interpretation.json'), reused_identical_facts_from='conditional_only')
                        write(target/'interpretation.json', reused)
                        result['insights'][a] = reused['insight']
                    else:
                        result['insights'][a] = interpret(slug, evidence, bundle, facts, target)
                    result['arm_status'][a] = 'applied' if read(target/'interpretation.json')['use_refinement'] else 'declined'
                except Exception:
                    result['arm_status'][a] = 'failed'
                    write(target/'error.json', dict(error=traceback.format_exc()))
            result.update(status='computed', measure_count=len(partitions),
                          split_measure_count=sum(len(p['leaves'])>1 for p in partitions),
                          compression_changed_count=sum(any(abs(a-l['mean'])>1e-9 for a,l in zip(c['reconstructed_means'],p['leaves'])) for p,c in zip(partitions,compressions)))
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
        write(folder/'error.json', dict(error=result['error']))
    write(folder/'branch.json', result)
    return result

class HyperparameterReplay(replay.ReplayAblation):
    run = TrajStoryAgent.run
    refine_initial = TrajStoryAgent.refine_initial
    refine_final = TrajStoryAgent.refine_final

    def __init__(self, llm, source, generator, **parameters):
        super().__init__(llm, 'hyperparameter', source, generator=generator)
        self.synthesizer.llm = self.llm
        def refiner(slug, evidence, bundle, profile, folder):
            frozen = self.source_run/'early'/evidence['question_id'] if Path(folder).parent.name=='early' else None
            return branch(slug, evidence, bundle, profile, folder, frozen=frozen, **parameters)
        self.refiner = refiner
