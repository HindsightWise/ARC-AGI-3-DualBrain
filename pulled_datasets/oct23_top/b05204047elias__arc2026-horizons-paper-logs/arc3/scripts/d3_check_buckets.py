# SPDX-License-Identifier: MIT-0
"""Synthetic policy checks, no game scores or additional game runs."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from arc3.run_local import load_agent_class
from arcengine import GameAction,GameState
cls=load_agent_class(ROOT/'arc3/agent/my_agent.py')
module=sys.modules[cls.__module__]

def bare(b):
    a=cls.__new__(cls)
    a.BUDGET_BUCKETS=b
    a.frame_processor=module.FrameProcessor()
    a.graph_explorer=module.GraphExplorer(n_groups=5)
    a._level=a._new_level(0)
    a.last_hashed_frame=a.last_action=None
    a.level_up=True
    a._last_xy=None
    a._skip_note=False
    return a

def observe(a,n=None):
    frame=np.zeros((64,64),dtype=np.uint8)
    if n is not None: frame[63,:n]=9
    a.last_hashed_frame=a.last_action=None
    a.level_up=True # graph rebuild, retain per-level reference
    a.choose_action([],SimpleNamespace(state=GameState.NOT_FINISHED,frame=[frame],available_actions=[1]))
    return a.last_hashed_frame

a=bare(4)
lengths=[64,60,49,48,47,32,31,16,15,0]
keys=[observe(a,n) for n in lengths]
assert len(set(keys[:4]))==1
assert keys[4]==keys[5] and keys[6]==keys[7] and keys[8]==keys[9]
assert len(set(keys))==4 and all(len(k)==32 for k in keys)
assert observe(bare(0))==observe(bare(4))
assert observe(bare(0),64)==observe(bare(0),32)
print(json.dumps({'backend':'synthetic','checks':'same bucket same hash; four bins; U32; no bar baseline hash; B0 masked bar invariant','passed':True,'lengths':lengths,'hashes':keys}))
