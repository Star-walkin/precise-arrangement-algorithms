"""Load the frozen offline H6 helpers in a separate import namespace."""
import importlib.util,sys
from common import H6,load,sha
def get():
    frozen=load(H6/'freeze_final.json')
    assert all(sha(p)==v for p,v in (frozen['sources']|frozen['reused_sources']).items())
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
    old_common=sys.modules.get('common');sys.path.insert(0,str(H6))
    try:
        c=module('_h7_frozen_h6_common',H6/'common.py');sys.modules['common']=c
        gallery=module('_h7_frozen_h6_gallery',H6/'run_h6.py')
        return gallery
    finally:
        sys.path.remove(str(H6))
        if old_common is not None:sys.modules['common']=old_common
        else:sys.modules.pop('common',None)
