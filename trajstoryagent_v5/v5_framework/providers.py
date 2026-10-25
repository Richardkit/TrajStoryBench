"""Provider settings and credentials without historical experiment imports."""
import json
import os
import sys
from pathlib import Path

MODELS = {
    'gpt': ('gpt-6-luna', 'https://api.openai.com/v1'),
    'qwen': ('qwen3.8-flash', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
}


def credentials(generator):
    """Load only the selected provider's key; never log credential contents."""
    name = {'gpt': 'OPENAI_API_KEY', 'qwen': 'DASHSCOPE_API_KEY'}[generator]
    if os.environ.get(name):
        return
    value = None
    state = Path(sys.prefix)/'conda-meta/state'
    if state.is_file():
        value = json.loads(state.read_text(encoding='utf-8')).get('env_vars', {}).get(name)
    if sys.platform == 'win32':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
                value = winreg.QueryValueEx(key, name)[0] or value
        except OSError:
            pass
    if not value:
        raise RuntimeError(name + ' is missing')
    os.environ[name] = value
