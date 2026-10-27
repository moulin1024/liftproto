#!/usr/bin/env python3
"""Static structural diff: lift a candidate (e.g. an optimised version) and compare
its structure with the reference structure, for ALL problem sizes.

  identical            same lifted term                      -> license: compiler
  IEEE-identical       equal modulo commutativity of + and *  -> license: compiler
  R-equivalent         z3 proves equality over the reals      -> license: statistical
  different / opaque   structure changed or not liftable      -> reject or needs evidence
"""
import json, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z3
from lift import ir
from lift.static.analysis import analyze
from lift.static import proofs


def r_equal(a, b):
    """a == b over the reals, wherever both are defined (all denominators non-zero)."""
    from lift.static.lifter import nodes
    try:
        Z = proofs.Z(int_names=('NX', 'NY', 'i', 'j', 'N'))
        dens = [Z.tr(x.b) != 0 for e in (a, b) for x in nodes(e) if isinstance(x, ir.Bin) and x.op == '/']
        return Z.valid(z3.And(dens) if dens else z3.BoolVal(True), Z.tr(a) == Z.tr(b))
    except Exception:
        return None


def compare(ref, cand):
    out = {}
    for n, rs in ref['summaries'].items():
        cs = cand['summaries'].get(n)
        if rs.term is None:
            out[n] = 'reference opaque'; continue
        if cs is None or cs.term is None:
            r = cs.reason if cs else 'missing'
            out[n] = (f'DIFFERENT SEMANTICS: {r}' if 'counterexample' in r else f'CANDIDATE NOT LIFTABLE: {r}'); continue
        if ir.pp_def(rs.term) == ir.pp_def(cs.term):
            fr = {k: str(v) for k, v in rs.footprint.items()}; fc = {k: str(v) for k, v in cs.footprint.items()}
            out[n] = 'identical' if fr == fc else f'SAME TERM, DIFFERENT FOOTPRINT {fc} vs {fr}'; continue
        if rs.kind != 'kernel':
            out[n] = 'different driver structure'; continue
        ri, ci = proofs.inline(rs.term), proofs.inline(cs.term)
        if set(ri) != set(ci):
            out[n] = 'DIFFERENT OUTPUTS'; continue
        if all(proofs.ckey(ri[k]) == proofs.ckey(ci[k]) for k in ri):
            out[n] = 'IEEE-identical (commutativity only)'; continue
        eq = [r_equal(ri[k], ci[k]) for k in ri]
        out[n] = 'R-equivalent (z3), float-different' if all(eq) else 'DIFFERENT SEMANTICS'
    return out


def verdict(d):
    v = list(d.values())
    if any(x.startswith(('DIFFERENT', 'SAME TERM')) for x in v): return 'REJECT (structure changed)'
    if any(x.startswith('CANDIDATE NOT LIFTABLE') for x in v): return 'NEEDS EVIDENCE (not statically liftable)'
    if any(x.startswith('R-equivalent') for x in v): return 'ACCEPT? license: statistical'
    return 'ACCEPT license: compiler'


if __name__ == '__main__':
    ref = analyze('fortran/cg.f', 'fortran/cgmain.f')
    sym_ref = proofs.symmetry(ref['summaries']['AMUL'].term)['symmetric']
    res = {}
    for f in sorted(glob.glob('out/mutants/*/cg.f')):
        name = f.split('/')[-2]
        if name == 'reference': continue
        cand = analyze(f, 'fortran/cgmain.f')
        d = compare(ref, cand)
        changed = {k: v for k, v in d.items() if v != 'identical'}
        sym = None
        if cand['summaries']['AMUL'].term is not None:
            sym = proofs.symmetry(cand['summaries']['AMUL'].term)['symmetric']
        res[name] = {'verdict': verdict(d), 'changed_units': changed, 'AMUL_symmetric': sym}
        print(f'{name:22s} {res[name]["verdict"]:42s} {changed}  symmetric={sym}')
    json.dump(res, open('out/structdiff.json', 'w'), indent=1)
