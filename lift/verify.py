"""Deterministic checker: structural facts + the verification ladder.

V1  random inputs, relative error            (R-semantics, Schwartz-Zippel style)
V2  production states captured from the legacy run
V3  order-preserving bitwise replay           (structural identity, incl. signed zeros)
V4  branch coverage (gcov, separate phase)
Multi-size: two agent-visible sizes + one checker-secret size.
"""
import hashlib
import numpy as np
from . import ir
from .probe import Ctx, _beq
from .shape import classify

REL_TOL = 1e-12


def _f(v):
    return np.atleast_1d(np.asarray(v, dtype=np.float64))


def compare(fo, io, keys):
    bit, err = True, 0.0
    for k in keys:
        a, b = _f(fo[k]), _f(io[k])
        if a.shape != b.shape:
            return False, np.inf
        bit &= bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))
        s = max(np.max(np.abs(a), initial=0.0), 1e-300)
        err = max(err, float(np.max(np.abs(a - b), initial=0.0) / s))
    return bit, err


def ir_globals(ctx):
    g = {}
    for blk, n, dt, shape in ctx.lib.common_items():
        if dt != np.float64:
            g[n] = int(ctx.lib.common(blk)[n])
            continue
        v = ctx.lib.common(blk)[n]
        g[n] = v[:ctx.N].copy() if shape else float(v)
    return g


def run_ir(ctx, term, fns, inputs):
    g = ir_globals(ctx)
    for k, v in inputs.items():
        if '.' in k:
            g[k.split('.')[1]] = v
    outer = {'NX': ctx.env['NX'], 'NY': ctx.env['NY'], 'N': ctx.N, '__globals__': g}
    from .probe import _ev
    args = {p: inputs[p] if p in inputs else _ev(ctx.spec['int_args'][p], ctx.env) for p in term.params}
    return ir.run(term, args, fns, outer)


# ----------------------------------------------------------------------------- facts
def facts(lib, uname, spec, nx, ny, rng):
    ctx = Ctx(lib, uname, spec, nx, ny)
    w, oob = ctx.write_set(rng)
    outkeys = [k for k in w]
    fl_out = [k for k in outkeys if k == '__result__' or k in ctx.keys()]
    r = ctx.read_set(rng, fl_out)
    out = {'write_set': w, 'read_set': r, 'oob': oob, 'degree': {}, 'homogeneity': {}, 'shape': {}}
    if spec.get('setup'):
        return out, ctx
    for k in r:
        out['degree'][k] = ctx.degree(rng, k, fl_out)
        out['homogeneity'][k] = ctx.homogeneity(rng, k, fl_out)
    ints = [ctx.env['NX'], ctx.env['NY']]
    for kin in r:
        if '.' in kin or not ctx.shape_of(kin):
            continue
        for kout in fl_out:
            if kout != '__result__' and not ctx.shape_of(kout):
                continue
            R = ctx.dependency(rng, kin, kout)
            out['shape'][f'{kin}->{kout}'] = classify(R, ints)
    return out, ctx


def operator_coefficients(ctx, kin, kout, offsets, others):
    """Exact coefficient fields c_(di,dj)[i,j] of a linear stencil operator."""
    A = ctx.jacobian_exact(kin, kout, others)
    nx, ny = ctx.env['NX'], ctx.env['NY']
    C = {}
    for (di, dj) in offsets:
        c = np.zeros(nx * ny)
        for k in range(nx * ny):
            i, j = k % nx + di, k // nx + dj
            if 0 <= i < nx and 0 <= j < ny:
                c[k] = A[k, j * nx + i]
        C[(di, dj)] = c
    return C, A


# ---------------------------------------------------------------------- ladder
class Digest:
    def __init__(self): self.h = hashlib.sha256()
    def add(self, outs, keys):
        for k in keys:
            self.h.update(_f(outs[k]).tobytes())
    def hex(self): return self.h.hexdigest()[:16]


def production_states(lib, nx, ny, ks=(1, 5, 20)):
    spec = {'arg_len': {'B': 'N', 'X': 'N', 'W': '4*N'}, 'int_args': {'N': 'N', 'ITER': '0'},
            'common_int': {'GRID.NX': 'NX', 'GRID.NY': 'NY'}, 'setup': ('SETUP', {'NXI': 'NX', 'NYI': 'NY'})}
    states = []
    for k in ks:
        spec['int_args']['MAXIT'] = str(k)
        ctx = Ctx(lib, 'CGSOLV', spec, nx, ny)
        n = ctx.N
        outs, _ = ctx.run({'B': np.ones(n), 'X': np.zeros(n), 'W': np.zeros(4 * n), 'TOL': 1e-12, 'RNORM': 0.0})
        W = outs['W']
        states.append({'R': W[:n], 'Z': W[n:2 * n], 'P': W[2 * n:3 * n], 'Q': W[3 * n:], 'X': outs['X']})
    return states


def ladder(lib, uname, spec, term, fns, sizes, seed, n_random=8, binding=None, digest=None, production=None):
    rec = {'sizes': [], 'oob': [], 'V1_max_rel_err': 0.0, 'V2_max_rel_err': None,
           'V3_bitwise': True, 'n_probes': 0}
    rng = np.random.default_rng(seed)
    keys = list(term.outs)
    for (nx, ny) in sizes:
        rec['sizes'].append([nx, ny])
        ctx = Ctx(lib, uname, spec, nx, ny)
        for r_ in range(n_random):                                  # V1 + V3
            inp = ctx.draw(rng)
            fo, oob = ctx.run(inp, check_reads=(r_ == 0))           # differential canaries once per size
            rec['oob'] += oob
            io = run_ir(ctx, term, fns, inp)
            bit, err = compare(fo, io, keys)
            rec['V1_max_rel_err'] = max(rec['V1_max_rel_err'], err)
            rec['V3_bitwise'] &= bit
            rec['n_probes'] += 1
            if digest: digest.add(fo, keys)
        if binding:                                                 # V2 + V3
            pspec = dict(spec, setup=('SETUP', {'NXI': 'NX', 'NYI': 'NY'}))
            pctx = Ctx(lib, uname, pspec, nx, ny)
            for st in production_states(lib, nx, ny):
                inp = pctx.draw(rng)
                inp.update({a: st[s] for a, s in binding.items()})
                fo, oob = pctx.run(inp)
                rec['oob'] += oob
                io = run_ir(pctx, term, fns, inp)
                bit, err = compare(fo, io, keys)
                rec['V2_max_rel_err'] = max(rec['V2_max_rel_err'] or 0.0, err)
                rec['V3_bitwise'] &= bit
                rec['n_probes'] += 1
                if digest: digest.add(fo, keys)
    for prod in production or []:                                # V2 from captured call arguments
        nx, ny = prod['size']
        pspec = dict(spec)
        if prod.get('setup'): pspec['setup'] = prod['setup']
        pctx = Ctx(lib, uname, pspec, nx, ny)
        inp = pctx.draw(rng)
        inp.update({k: v for k, v in prod['args'].items() if k in inp})
        fo, oob = pctx.run(inp)
        rec['oob'] += oob
        io = run_ir(pctx, term, fns, inp)
        bit, err = compare(fo, io, keys)
        rec['V2_max_rel_err'] = max(rec['V2_max_rel_err'] or 0.0, err)
        rec['V3_bitwise'] &= bit
        rec['n_probes'] += 1
        if digest: digest.add(fo, keys)
    rec['oob'] = sorted(set(rec['oob']))
    rec['status'] = status(rec)
    return rec


def status(rec):
    if rec['oob']:
        return 'REFUTED (out-of-bounds access: length hypothesis wrong)'
    if rec['V1_max_rel_err'] > REL_TOL or (rec['V2_max_rel_err'] or 0) > REL_TOL:
        return 'REFUTED'
    return 'BITWISE-EQUIVALENT' if rec['V3_bitwise'] else 'R-EQUIVALENT (float-different)'


def targeted(lib, uname, term, fns, probes):
    """Run agent-proposed targeted probes (explicit argument values)."""
    res = []
    for p in probes:
        fo, oob = lib.call(uname, p)
        n = p.get('N', 0)
        io = ir.run(term, {k: p[k] for k in term.params}, fns, {'NX': 0, 'NY': 0, 'N': n, '__globals__': {}})
        bit, err = compare(fo, io, list(term.outs))
        res.append({'probe': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in p.items()},
                    'bitwise': bit, 'rel_err': err, 'oob': oob,
                    'fortran': {k: _f(fo[k]).tolist() for k in term.outs},
                    'lifted': {k: _f(io[k]).tolist() for k in term.outs}})
    return res
