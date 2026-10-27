#!/usr/bin/env python3
"""CLI for agents with a shell (e.g. Claude Code):

    python liftctl.py init   WS  src.f [main.f]
    python liftctl.py WS TOOL '{"json": "args"}'      # see: python liftctl.py WS help
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift import tools
from lift.workspace import Workspace

if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == 'init':
        ws = Workspace.create(a[1], a[2], a[3] if len(a) > 3 else None)
        print(json.dumps(ws.status(), indent=1)); sys.exit(0)
    if len(a) < 2 or a[1] == 'help':
        print(json.dumps(tools.TOOLS, indent=1)); sys.exit(0)
    args = json.loads(a[2]) if len(a) > 2 else {}
    r = tools.call(a[0], a[1], args)
    print(r if isinstance(r, str) else json.dumps(r, indent=1))
