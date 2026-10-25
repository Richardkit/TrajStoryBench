"""Shared credentials and file utilities; never imports historical experiment coordinators."""
import hashlib,os,sys
from pathlib import Path
from .evaluation import read,write

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_files():
    return sorted(p for p in Path(__file__).parent.iterdir() if p.is_file() and p.suffix in ('.py','.json'))

def credentials(names):
    state=Path(sys.prefix)/'conda-meta/state'
    values=read(state).get('env_vars',{}) if state.exists() else {}
    for name in names:
        if os.environ.get(name):continue
        value=values.get(name)
        if sys.platform=='win32':
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as key:
                    value=winreg.QueryValueEx(key,name)[0] or value
            except OSError:pass
        if not value:raise RuntimeError(name+' is missing; configure it before running')
        os.environ[name]=value
