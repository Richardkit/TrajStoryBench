"""Record provider usage for both streamed analysis and nonstream final synthesis."""
import json
from pathlib import Path
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI


class UsageChatOpenAI(ChatOpenAI):
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs['stream_options'] = {'include_usage': True}
        yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)


class ProviderUsageRecorder(BaseCallbackHandler):
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.calls = []
        self.indices = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.indices[str(run_id)] = len(self.calls)
        self.calls.append(dict(call=len(self.calls)+1, run_id=str(run_id), state='started', usage=None))
        self._save()

    def _usage(self, response):
        for generations in response.generations:
            for generation in generations:
                usage = getattr(generation.message, 'usage_metadata', None)
                if usage and usage.get('input_tokens') is not None and usage.get('output_tokens') is not None:
                    return dict(prompt_tokens=usage['input_tokens'], completion_tokens=usage['output_tokens'],
                                total_tokens=usage.get('total_tokens',usage['input_tokens']+usage['output_tokens']))
        usage = (response.llm_output or {}).get('token_usage')
        if usage and all(usage.get(k) is not None for k in ('prompt_tokens','completion_tokens','total_tokens')):
            return {k:usage[k] for k in ('prompt_tokens','completion_tokens','total_tokens')}
        return None

    def on_llm_end(self, response, *, run_id, **kwargs):
        row = self.calls[self.indices[str(run_id)]]
        row.update(state='returned', usage=self._usage(response))
        self._save()

    def on_llm_error(self, error, *, run_id, **kwargs):
        if str(run_id) in self.indices:
            self.calls[self.indices[str(run_id)]].update(state='error', error_type=type(error).__name__)
        self._save()

    def summary(self):
        known = [r['usage'] for r in self.calls if r['usage'] is not None]
        network_failed = sum(r.get('error_type') == 'ConsecutiveNetworkFailures' for r in self.calls)
        return dict(source='provider_reported_usage', model_calls_started=len(self.calls),
                    requests_used=len(self.calls)-network_failed,
                    calls_with_usage=len(known), missing_usage_calls=len(self.calls)-len(known),
                    complete=bool(self.calls) and len(known)==len(self.calls),
                    **{k:sum(u[k] for u in known) for k in ('prompt_tokens','completion_tokens','total_tokens')})

    def _save(self):
        self.output_dir.mkdir(parents=True,exist_ok=True)
        (self.output_dir/'model_usage.json').write_text(json.dumps(dict(summary=self.summary(),calls=self.calls),indent=2),encoding='utf8')
