"""End-to-end: lift a legacy F77 file into the functional IR and certify it.

  python run_lift.py fortran/cg.f            # full pipeline -> out/
Phases (each a separate process so gcov data is flushed):
  probe   : facts + verification ladder on the -O2 build     (round 1)
  cov     : replay the same probes on an -O0 --coverage build (V4 + flag invariance)
  round2  : agent's targeted probes for uncovered branches, revisions
  report  : certificates.json + report.txt
"""
import json, os, pickle, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift import harness, ir, verify
from lift.probe import Ctx
import hypotheses.cg_agent as agent

ORDER = ['DDOT', 'DAXPY', 'DCOPY', 'XPAY', 'PCOND', 'AMUL', 'CGSOLV']
SEED = 20261027
SECRET_SEED = 7                      # checker-private; logged in the certificate after the fact
OUT = os.path.abspath('out')


def sizes():
    r = np.random.default_rng(SECRET_SEED)
    return [(7, 5), (6, 8), tuple(int(v) for v in r.integers(9, 17, size=2))]


def verification(lib, fns, digests=None, include_targeted=False):
    res = {}
    for k, u in enumerate(ORDER):
        d = verify.Digest() if digests is not None else None
        res[u] = verify.ladder(lib, u, agent.SPECS[u], fns[u], fns, sizes(), SEED + k,
                               binding=agent.PRODUCTION_BINDING.get(u), digest=d)
        if include_targeted and u in agent.TARGETED_PROBES:
            verify.targeted(lib, u, fns[u], fns, agent.TARGETED_PROBES[u])
        if d is not None:
            digests[u] = d.hex()
    return res


def phase_probe(src):
    os.makedirs(OUT, exist_ok=True)
    so = os.path.join(OUT, 'build', 'libcg.so')
    binfo = harness.build(src, so)
    lib = harness.FLib(so, src)
    F = {}
    for k, u in enumerate(ORDER):
        f, ctx = verify.facts(lib, u, agent.SPECS[u], 7, 5, np.random.default_rng(SEED + 100 + k))
        F[u] = f
    # operator structure of AMUL: exact coefficient fields + agent explanation + properties
    amul = {}
    for label, use_setup in (('random_coefficients', False), ('SETUP_geometry', True)):
        for (nx, ny) in sizes():
            spec = dict(agent.SPECS['AMUL'])
            if use_setup:
                spec['setup'] = ('SETUP', {'NXI': 'NX', 'NYI': 'NY'})
            ctx = Ctx(lib, 'AMUL', spec, nx, ny)
            others = ctx.draw(np.random.default_rng(SEED + nx * ny))
            offs = F['AMUL']['shape']['X->Y'][1]['offsets']
            C, A = verify.operator_coefficients(ctx, 'X', 'Y', [tuple(o) for o in offs], others)
            g = verify.ir_globals(ctx)
            for kk, v in others.items():
                if '.' in kk: g[kk.split('.')[1]] = v
            env = {'NX': nx, 'NY': ny, 'N': nx * ny, **g}
            explained = all(
                np.array_equal(C[o].view(np.uint64)[mask := (C[o] != 0) | (o == (0, 0))],
                               np.asarray(ir.ev(agent.COEFFS_AMUL[o], env, {}), dtype=float).view(np.uint64)[mask])
                for o in C)
            sym = bool(np.array_equal(A, A.T))          # exact equality (structural zeros' signs ignored)
            diag, off = np.diag(A), np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))
            rel = (diag - off) / np.maximum(np.abs(diag), 1e-300)
            rec = amul.setdefault(label, {'coefficients_explained_bitwise': True, 'symmetric_exact': True,
                                          'diag_positive': True, 'weakly_diag_dominant_to_roundoff': True,
                                          'strictly_dominant_rows_exist': True, 'min_rel_dominance_margin': 1.0,
                                          'sizes': []})
            rec['sizes'].append([nx, ny])
            rec['coefficients_explained_bitwise'] &= bool(explained)
            rec['symmetric_exact'] &= sym
            rec['diag_positive'] &= bool(np.all(diag > 0))
            rec['weakly_diag_dominant_to_roundoff'] &= bool(np.all(rel >= -16 * np.finfo(float).eps))
            rec['strictly_dominant_rows_exist'] &= bool(np.any(rel > 1e-8))
            rec['min_rel_dominance_margin'] = min(rec['min_rel_dominance_margin'], float(rel.min()))
    s = amul['SETUP_geometry']
    s['SPD_certificate'] = bool(s['symmetric_exact'] and s['diag_positive'] and s['weakly_diag_dominant_to_roundoff']
                                and s['strictly_dominant_rows_exist']) and \
        'symmetric + positive diagonal + irreducibly diagonally dominant (5-point graph connected) => SPD (over R)'
    digests = {}
    V = verification(lib, agent.TERMS, digests)
    json.dump({'build': binfo, 'facts': F, 'amul_operator': amul, 'round1': V, 'digests_O2': digests,
               'sizes': sizes()}, open(os.path.join(OUT, 'round1.json'), 'w'), indent=1, default=_j)


def phase_cov(src, tag, include_targeted):
    covdir = os.path.join(OUT, f'cov_{tag}')
    so = harness.build_cov(src, covdir)
    lib = harness.FLib(so, src)
    terms = dict(agent.TERMS); terms.update(agent.REVISIONS if include_targeted else {})
    digests = {}
    verification(lib, terms, digests, include_targeted)
    json.dump(digests, open(os.path.join(covdir, 'digests_O0.json'), 'w'))


def phase_round2(src, gaps):
    so = os.path.join(OUT, 'build', 'libcg.so')
    lib = harness.FLib(so, src)
    out = {}
    for u, g in gaps.items():
        if u not in agent.TARGETED_PROBES:
            continue
        r = {'uncovered': g, 'targeted_vs_round1': verify.targeted(lib, u, agent.TERMS[u], agent.TERMS,
                                                                   agent.TARGETED_PROBES[u])}
        if not all(t['bitwise'] for t in r['targeted_vs_round1']) and u in agent.REVISIONS:
            fns = dict(agent.TERMS); fns[u] = agent.REVISIONS[u]
            r['revision'] = ir.pp_def(agent.REVISIONS[u])
            r['targeted_vs_revision'] = verify.targeted(lib, u, fns[u], fns, agent.TARGETED_PROBES[u])
            r['ladder_revision'] = verify.ladder(lib, u, agent.SPECS[u], fns[u], fns, sizes(),
                                                 SEED + ORDER.index(u), binding=agent.PRODUCTION_BINDING.get(u))
            r['driver_with_revision'] = verify.ladder(lib, 'CGSOLV', agent.SPECS['CGSOLV'], fns['CGSOLV'], fns,
                                                      sizes(), SEED + ORDER.index('CGSOLV'))
        out[u] = r
    return out


def _j(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, tuple): return list(o)
    raise TypeError(type(o))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 'fortran/cg.f'
    if len(sys.argv) > 2:                      # sub-phase invocation
        ph = sys.argv[2]
        if ph == 'probe': phase_probe(src)
        if ph == 'cov1': phase_cov(src, 'r1', False)
        if ph == 'cov2': phase_cov(src, 'r2', True)
        return
    me = [sys.executable, os.path.abspath(__file__), src]
    for ph in ('probe', 'cov1'):
        subprocess.run(me + [ph], check=True)
    units = harness.FLib(os.path.join(OUT, 'build', 'libcg.so'), src).units
    gaps1 = harness.gcov_partial(src, os.path.join(OUT, 'cov_r1'), units)
    r2 = phase_round2(src, gaps1)
    subprocess.run(me + ['cov2'], check=True)
    gaps2 = harness.gcov_partial(src, os.path.join(OUT, 'cov_r2'), units)
    R1 = json.load(open(os.path.join(OUT, 'round1.json')))
    d0 = json.load(open(os.path.join(OUT, 'cov_r1', 'digests_O0.json')))
    R1['flag_invariance_O2_vs_O0'] = {u: R1['digests_O2'][u] == d0[u] for u in ORDER}
    R1['coverage_round1'] = gaps1
    R1['round2'] = r2
    R1['coverage_round2'] = gaps2
    R1['secret_seed'] = SECRET_SEED
    R1['lifted_terms'] = {u: ir.pp_def((agent.REVISIONS.get(u) or agent.TERMS[u])) for u in ORDER}
    json.dump(R1, open(os.path.join(OUT, 'certificates.json'), 'w'), indent=1, default=_j)
    print('wrote', os.path.join(OUT, 'certificates.json'))


if __name__ == '__main__':
    main()
