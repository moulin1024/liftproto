"""AGENT OUTPUT -- untrusted.

Everything in this file is what the LLM agent proposed after reading cg.f.
Nothing here is believed by the framework until the deterministic checker
has verified it.  (In this prototype Claude played the agent by hand.)
"""
from lift.ir import (Var, Lit, Un, Shift, Cmp, BoolOp, Where, Fold, App, Concat,
                     Let, While, Def, I, J, NX, NY)

V = Var

# ----------------------------------------------------------------------------
# 1. Probing specs: array lengths, integer arguments, COMMON active lengths.
#    (Array lengths are checked by NaN canaries; a wrong guess is caught.)
# ----------------------------------------------------------------------------
GRID_COMMON = {'GRID.NX': 'NX', 'GRID.NY': 'NY'}
SPECS = {
    'AMUL':  dict(arg_len={'X': 'N', 'Y': 'N'}, int_args={'N': 'N'}, common_int=GRID_COMMON,
                  common_len={'GEOM.CK': 'N', 'GEOM.DG': 'N'}),
    'PCOND': dict(arg_len={'R': 'N', 'Z': 'N'}, int_args={'N': 'N'}, common_int=GRID_COMMON,
                  common_len={'GEOM.CK': 'N', 'GEOM.DG': 'N'}),
    'DDOT':  dict(arg_len={'DX': 'N', 'DY': 'N'}, int_args={'N': 'N'}),
    'DAXPY': dict(arg_len={'DX': 'N', 'DY': 'N'}, int_args={'N': 'N'}),
    'DCOPY': dict(arg_len={'DX': 'N', 'DY': 'N'}, int_args={'N': 'N'}),
    'XPAY':  dict(arg_len={'Z': 'N', 'P': 'N'}, int_args={'N': 'N'}),
    'CGSOLV': dict(arg_len={'B': 'N', 'X': 'N', 'W': '4*N'},
                   int_args={'N': 'N', 'MAXIT': '1000', 'ITER': '0'},
                   common_int=GRID_COMMON, setup=('SETUP', {'NXI': 'NX', 'NYI': 'NY'}),
                   fixed={'TOL': 1e-10}),
}

# ----------------------------------------------------------------------------
# 2. Lifted terms (round 1).
# ----------------------------------------------------------------------------
cx = V('HX2I') * 0.5
cy = V('HY2I') * 0.5
X, CK, DG = V('X'), V('CK'), V('DG')
S0 = DG * X
S1 = Where(Cmp('>', I, Lit(1)),  V('S0') - cx * (CK + Shift(CK, -1, 0)) * Shift(X, -1, 0), V('S0'))
S2 = Where(Cmp('<', I, NX),      V('S1') - cx * (CK + Shift(CK, +1, 0)) * Shift(X, +1, 0), V('S1'))
S3 = Where(Cmp('>', J, Lit(1)),  V('S2') - cy * (CK + Shift(CK, 0, -1)) * Shift(X, 0, -1), V('S2'))
S4 = Where(Cmp('<', J, NY),      V('S3') - cy * (CK + Shift(CK, 0, +1)) * Shift(X, 0, +1), V('S3'))

TERMS = {
    'AMUL':  Def('AMUL', ['X'], [Let('S0', S0), Let('S1', S1), Let('S2', S2), Let('S3', S3)],
                 {'Y': S4}),
    'PCOND': Def('PCOND', ['R'], [], {'Z': V('R') / DG}),
    # manual unroll-by-5 is lifted to a plain sequential fold: unrolling is the
    # compiler's job (license: compiler), not part of the semantics.
    'DDOT':  Def('DDOT', ['DX', 'DY'], [], {'__result__': Fold(V('DX') * V('DY'))}),
    'DAXPY': Def('DAXPY', ['DA', 'DX', 'DY'], [], {'DY': V('DY') + V('DA') * V('DX')}),
    'DCOPY': Def('DCOPY', ['DX'], [], {'DY': V('DX')}),
    'XPAY':  Def('XPAY', ['Z', 'BETA', 'P'], [], {'P': V('Z') + V('BETA') * V('P')}),
}

# The driver: GOTO loop -> while_loop; hand-partitioned W -> named vectors.
nrm = lambda v: Un('sqrt', App('DDOT', (v, v)))
TERMS['CGSOLV'] = Def(
    'CGSOLV', ['B', 'X', 'TOL'],
    [Let('Q', App('AMUL', (V('X'),))),
     Let('R', V('B') - V('Q')),
     Let('Z', App('PCOND', (V('R'),))),
     Let('P', App('DCOPY', (V('Z'),))),
     Let('RHO', App('DDOT', (V('R'), V('Z')))),
     Let('BNRM', nrm(V('B'))),
     Let('ITER', Lit(0.0)),
     Let('RNORM', nrm(V('R'))),
     While(BoolOp('and', BoolOp('not', Cmp('<=', V('RNORM'), V('TOL') * V('BNRM'))),
                  Cmp('<', V('ITER'), Lit(1000.0))),
           [Let('ITER', V('ITER') + 1.0),
            Let('Q', App('AMUL', (V('P'),))),
            Let('ALPHA', V('RHO') / App('DDOT', (V('P'), V('Q')))),
            Let('X', App('DAXPY', (V('ALPHA'), V('P'), V('X')))),
            Let('R', App('DAXPY', (-V('ALPHA'), V('Q'), V('R')))),
            Let('Z', App('PCOND', (V('R'),))),
            Let('RHONEW', App('DDOT', (V('R'), V('Z')))),
            Let('BETA', V('RHONEW') / V('RHO')),
            Let('RHO', V('RHONEW')),
            Let('P', App('XPAY', (V('Z'), V('BETA'), V('P')))),
            Let('RNORM', nrm(V('R')))])],
    {'X': V('X'), 'W': Concat((V('R'), V('Z'), V('P'), V('Q'))),
     'ITER': V('ITER'), 'RNORM': V('RNORM')})

# Which production vector feeds which kernel argument (for V2 probes).
PRODUCTION_BINDING = {
    'AMUL': {'X': 'P'}, 'PCOND': {'R': 'R'}, 'DDOT': {'DX': 'R', 'DY': 'Z'},
    'DAXPY': {'DX': 'P', 'DY': 'R'}, 'DCOPY': {'DX': 'Z'}, 'XPAY': {'Z': 'Z', 'P': 'P'},
}

# ----------------------------------------------------------------------------
# 3. Round 2: after the checker reported uncovered branches, the agent
#    proposes targeted probes, and (if refuted) revised terms.
# ----------------------------------------------------------------------------
import numpy as np
TARGETED_PROBES = {
    # hit 'IF (DA.EQ.0.0D0) RETURN' with signed zeros in DY
    'DAXPY': [dict(N=6, DA=0.0, DX=np.array([1., 2, 3, 4, 5, 6]),
                   DY=np.array([-0.0, 1., -0.0, 2., 3., -0.0])),
              dict(N=3, DA=0.5, DX=np.array([1., 2, 3]), DY=np.array([1., 1, 1])),   # N < 4 path
              dict(N=0, DA=0.5, DX=np.zeros(0), DY=np.zeros(0))],                     # N <= 0 path
    'DDOT':  [dict(N=3, DX=np.array([1., 2, 3]), DY=np.array([.1, .2, .3])),          # N < 5 path
              dict(N=0, DX=np.zeros(0), DY=np.zeros(0))],                              # N <= 0 path
}
REVISIONS = {
    'DAXPY': Def('DAXPY', ['DA', 'DX', 'DY'], [],
                 {'DY': Where(Cmp('==', V('DA'), Lit(0.0)), V('DY'), V('DY') + V('DA') * V('DX'))}),
}

# ----------------------------------------------------------------------------
# 4. Coefficient explanation for the linear operator AMUL (A = sum over
#    offsets of coefficient fields).  Order of IEEE operations as in source.
# ----------------------------------------------------------------------------
COEFFS_AMUL = {
    (0, 0): DG,
    (-1, 0): -(cx * (CK + Shift(CK, -1, 0))),
    (+1, 0): -(cx * (CK + Shift(CK, +1, 0))),
    (0, -1): -(cy * (CK + Shift(CK, 0, -1))),
    (0, +1): -(cy * (CK + Shift(CK, 0, +1))),
}
