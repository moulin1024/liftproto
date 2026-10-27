"""Coverage replay (separate process so gcov data is flushed at exit):
re-runs every verification probe and every recorded targeted probe on an
-O0 --coverage build of the legacy code."""
import os, sys
import numpy as np
from . import harness, verify
from .workspace import Workspace


def main(path):
    ws = Workspace(path)
    covdir = os.path.join(ws.path, 'cov')
    so = harness.build_cov(ws.state['src'], covdir)
    ws._lib = harness.FLib(so, ws.state['src'])
    fns = ws.fns()
    prod = ws.capture_production()
    for k, n in enumerate(ws.state['units']):
        if not ws.spec_complete(n): continue
        u = ws.state['units'][n]
        if u['term'] and ws.missing_callees(n, fns):
            continue
        if u['term']:
            verify.ladder(ws.lib, n, ws.spec(n), fns[n], fns, ws.sizes(), 1000 + k, production=prod.get(n))
        else:
            from .probe import Ctx
            for (nx, ny) in ws.sizes():
                ctx = Ctx(ws.lib, n, ws.spec(n), nx, ny)
                for s in range(8): ctx.run(ctx.draw(np.random.default_rng(s)))
            for p in prod.get(n, []):
                ctx = Ctx(ws.lib, n, ws.spec(n), *p['size'])
                d = ctx.draw(np.random.default_rng(0)); d.update({a: v for a, v in p['args'].items() if a in d})
                ctx.run(d)
    for n, probes in ws.state['targeted'].items():
        for p in probes:
            try:
                ws.exec_probe(n, p['inputs'], p.get('size'))
            except Exception as ex:
                print('probe failed', n, ex, file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv[1])
