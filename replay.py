#!/usr/bin/env python3
"""Replay the agent actions of a recorded session on a fresh workspace:
    python replay.py OLD_WS NEW_WS
Sessions are therefore reproducible evidence, not just transcripts."""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift import tools
from lift.workspace import Workspace

old, new = sys.argv[1], sys.argv[2]
st = json.load(open(os.path.join(old, 'state.json')))
shutil.rmtree(new, ignore_errors=True)
Workspace.create(new, st['src'], st['main'], secret_seed=st['secret_seed'])
for line in open(os.path.join(old, 'log.jsonl')):
    e = json.loads(line)
    if e['actor'] != 'agent' or e['tool'] in ('status', 'worklist', 'show', 'structure', 'dsl_help'):
        continue
    r = tools.call(new, e['tool'], e['args'])
    print(e['tool'], json.dumps(r, default=str)[:150])
tools.call(new, 'structure', {})
print('worklist:', tools.call(new, 'worklist', {}))
