"""Adversarial tests of the LLM channel: every malformed, wrong or hostile
proposal must be rejected by the checker (run on a scratch workspace)."""
import json, os, shutil, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lift import tools
from lift.workspace import Workspace

WS = os.path.join(ROOT, 'ws_adversarial')


def T(tool, args):
    return tools.call(WS, tool, args, actor='red-team')


def main():
    shutil.rmtree(WS, ignore_errors=True)
    Workspace.create(WS, os.path.join(ROOT, 'fortran/pcg.f'), os.path.join(ROOT, 'fortran/pcgmain.f'), secret_seed=11)
    cases = []
    def check(name, r, expect_reject=True):
        rejected = ('error' in r) or (r.get('accepted') is False) or (r.get('match') is False) or (r.get('consistent') is False)
        ok = rejected == expect_reject
        cases.append((name, ok, json.dumps(r, default=str)[:160]))

    check('spec: code injection in expression', T('propose_spec', {'unit': 'DNRM2', 'arg_len': {'X': '__import__("os").system("id")'}}))
    check('spec: too short (N-1) -> canary', T('propose_spec', {'unit': 'DNRM2', 'arg_len': {'X': 'N-1'}, 'int_args': {'INCX': '1'}}))
    check('spec: correct', T('propose_spec', {'unit': 'DNRM2', 'arg_len': {'X': 'N'}, 'int_args': {'INCX': '1'}}), expect_reject=False)
    check('term: python in DSL', T('propose_term', {'unit': 'DNRM2', 'term': 'DNRM2 = λ(X).\n    in  {__result__ = __import__("os")}'}))
    check('term: wrong math (L1 norm)', T('propose_term', {'unit': 'DNRM2', 'term': 'DNRM2 = λ(X).\n    in  {__result__ = foldl(+, 0, abs(X))}'}))
    check('primitive: wrong contract (L1 norm)', T('register_primitive', {'unit': 'DNRM2', 'relation': 'DNRM2 = λ(X).\n    in  {__result__ = foldl(+, 0, abs(X))}', 'properties': ['nonnegative']}))
    check('primitive: false property (homogeneous:2)', T('register_primitive', {'unit': 'DNRM2', 'relation': 'DNRM2 = λ(X).\n    in  {__result__ = sqrt(foldl(+, 0, X * X))}', 'properties': ['homogeneous:2']}))
    check('primitive: self-referential contract', T('register_primitive', {'unit': 'DNRM2', 'relation': 'DNRM2 = λ(X).\n    in  {__result__ = DNRM2(X)}'}))
    check('primitive: tolerance too loose', T('register_primitive', {'unit': 'DNRM2', 'relation': 'DNRM2 = λ(X).\n    in  {__result__ = sqrt(foldl(+, 0, X * X))}', 'tol': 0.1}))
    check('term: replace static DDOT by pairwise-sum-free wrong order',
          T('propose_term', {'unit': 'DDOT', 'term': 'DDOT = λ(DX, DY).\n    in  {__result__ = foldl(+, 0, DY * DX) * 1.0000001}'}))
    check('template: wrong algorithm (steepest descent: p = z)', T('propose_template', {'unit': 'CGSOLV', 'name': 'steepest descent',
          'bindings': {'A': 'AMUL'}, 'template': 'SD = λ(b, x, tol, maxit).\n    let q = A(x)\n    in  {X = x}'}))
    check('probe: oversize grid', T('run_probe', {'unit': 'CGSOLV', 'size': [1000, 1000], 'inputs': {}}))
    for name, ok, r in cases:
        print(('PASS ' if ok else 'FAIL ') + name.ljust(55) + r)
    print(f'{sum(ok for _, ok, _ in cases)}/{len(cases)} passed')
    return all(ok for _, ok, _ in cases)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
