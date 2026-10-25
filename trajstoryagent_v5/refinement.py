"""Formal v5 refinement implementation."""
import traceback
from trajstoryagent_v5.export_support import (
    ARMS, read, write, observations,
)
from trajstoryagent_v5.conditional_math import conditional_partition, render_facts
from trajstoryagent_v5.compression_math import compress_with_exceptions
from trajstoryagent_v5.interpretation import interpret

COST_MODEL = 'name_cost_only_no_fixed_value_cost'
SCHEMA_VERSION = 2

def run_branch(slug, evidence, bundle, profile, directory, *, min_support=1):
    checkpoint = directory/'branch.json'
    if checkpoint.exists():
        cached = read(checkpoint)
        if (cached.get('compression_cost_model') != COST_MODEL
                or cached.get('schema_version') != SCHEMA_VERSION
                or cached.get('min_support') != min_support):
            raise ValueError('Refusing to reuse a branch with a different compression cost model, schema or minimum support')
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    write(directory/'original_evidence.json', evidence)
    result = {'question_id': evidence['question_id'], 'question': evidence['question'],
              'original_insight': evidence['insight'], 'status': 'pending', 'compression_cost_model': COST_MODEL,
              'schema_version': SCHEMA_VERSION, 'min_support': min_support,
              'insights': {a: evidence['insight'] for a in ARMS}, 'arm_status': {a:'fallback' for a in ARMS}}
    try:
        obs = observations(slug, evidence, bundle, profile, directory)
        if not obs['measures']:
            result.update(status='unsupported', reason=obs.get('unsupported_reason'))
        else:
            partitions = [conditional_partition(m, min_support=min_support) for m in obs['measures']]
            compressions = [compress_with_exceptions(p) for p in partitions]
            write(directory/'mathematical_certificate.json', {'partitions':partitions, 'compressions':compressions, 'compression_cost_model': COST_MODEL})
            for arm in ARMS:
                folder = directory/arm
                folder.mkdir(exist_ok=True)
                facts = [fact for p,c in zip(partitions,compressions)
                         for fact in render_facts(p, c if arm=='conditional_compressed' else None)]
                write(folder/'facts.json', facts)
                try:
                    first=directory/'conditional_only'
                    if (arm=='conditional_compressed' and (first/'interpretation.json').exists()
                            and read(first/'facts.json')==facts):
                        reused={**read(first/'interpretation.json'),'reused_identical_facts_from':'conditional_only'}
                        write(folder/'interpretation.json',reused)
                        result['insights'][arm]=reused['insight']
                    else:
                        result['insights'][arm] = interpret(slug, evidence, bundle, facts, folder)
                    result['arm_status'][arm] = 'applied' if read(folder/'interpretation.json')['use_refinement'] else 'declined'
                except Exception:
                    write(folder/'error.json', {'traceback':traceback.format_exc()})
                    result['arm_status'][arm] = 'failed'
            result.update(status='computed', measure_count=len(partitions),
                split_measure_count=sum(len(p['leaves'])>1 for p in partitions),
                compression_changed_count=sum(any(abs(a-l['mean'])>1e-9 for a,l in zip(c['reconstructed_means'],p['leaves']))
                                              for p,c in zip(partitions,compressions)))
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
    write(checkpoint, result)
    return result

