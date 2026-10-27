"""Structural proofs on lifted terms (all problem sizes).

  linear_terms     A(x) = sum_s [guard_s] c_s * x[shift s]       (exact decomposition of the term)
  symmetry         c_s(p) == c_-s(p+s) modulo commutativity of + and *  => IEEE-exact symmetry
  jacobi           preconditioner term == x / (diagonal coefficient of A)
  interval         interval arithmetic on IR (sin/cos in [-1,1], ...)
  spd              z3 (nonlinear real arithmetic): positive diagonal + diagonal dominance
                   at every grid point, strict at the boundary  => SPD over R
  degree           polynomial degree per parameter by abstract interpretation, with
                   R-collapse of IEEE-only branches (Where(p == c, a, b) with b[p:=c] == a over R)
"""
import math
import z3
from .. import ir
from ..ir import Var, Lit, Bin, Un, Shift, Idx, Cmp, BoolOp, Where, Fold, App, Concat, Let
from .lifter import key, nodes, rebuild, _children


def inline(defn, abstract_outputs=False):
    """Substitute lets.  With abstract_outputs, lets that ARE outputs (e.g. a COMMON field
    or mesh factor) stay symbolic under the output's name."""
    env = {}
    if abstract_outputs:
        for k, v in defn.outs.items():
            if isinstance(v, Var) and any(isinstance(l, Let) and l.name == v.name for l in defn.body):
                env[v.name] = Var(k)
    def sub(e):
        return rebuild(e, lambda x: env[x.name] if isinstance(x, Var) and x.name in env else None)
    for l in defn.body:
        if not isinstance(l, Let): raise ValueError('loop in kernel term')
        if l.name in env and abstract_outputs: continue
        env[l.name] = sub(l.e)
    return {k: sub(v) for k, v in defn.outs.items()}


# ---------------------------------------------------------------- linear terms
def linear_terms(e, x):
    """-> list of (shift (di,dj), guards [(cond, polarity)], coefficient) or raise ValueError."""
    def has_x(t): return any(isinstance(n, Var) and n.name == x for n in nodes(t))
    def go(t):
        if isinstance(t, Var) and t.name == x: return [((0, 0), [], None)]
        if isinstance(t, Shift) and isinstance(t.a, Var) and t.a.name == x: return [((t.di, t.dj), [], None)]
        if not has_x(t): raise ValueError('affine (non-homogeneous) part')
        if isinstance(t, Bin) and t.op in '+-':
            a, b = go(t.a), go(t.b)
            if t.op == '-': b = [(s, g, Un('neg', c) if c is not None else Un('neg', Lit(1.0))) for s, g, c in b]
            return a + b
        if isinstance(t, Bin) and t.op == '*':
            if not has_x(t.a): return [(s, g, t.a if c is None else Bin('*', t.a, c)) for s, g, c in go(t.b)]
            if not has_x(t.b): return [(s, g, t.b if c is None else Bin('*', c, t.b)) for s, g, c in go(t.a)]
            raise ValueError('nonlinear product')
        if isinstance(t, Bin) and t.op == '/' and not has_x(t.b):
            return [(s, g, Bin('/', c if c is not None else Lit(1.0), t.b)) for s, g, c in go(t.a)]
        if isinstance(t, Where):
            if has_x(t.c): raise ValueError('branch on x')
            return [(s, g + [(t.c, True)], c) for s, g, c in go(t.a)] + \
                   [(s, g + [(t.c, False)], c) for s, g, c in go(t.b)]
        if isinstance(t, Un) and t.op == 'neg':
            return [(s, g, Un('neg', c if c is not None else Lit(1.0))) for s, g, c in go(t.a)]
        raise ValueError(f'not linear in {x}: {type(t).__name__}')
    return [(s, g, c if c is not None else Lit(1.0)) for s, g, c in go(e)]


def ckey(e):
    """canonical key modulo commutativity of + and * (NOT associativity: IEEE-exact)."""
    if isinstance(e, Bin) and e.op in '+*':
        a, b = sorted([ckey(e.a), ckey(e.b)])
        return f'({a}{e.op}{b})'
    if isinstance(e, Bin): return f'({ckey(e.a)}{e.op}{ckey(e.b)})'
    if isinstance(e, Un): return f'{e.op}({ckey(e.a)})'
    if isinstance(e, Where): return f'ite({ckey(e.c)},{ckey(e.a)},{ckey(e.b)})'
    return key(e)


def shift_expr(e, s, grid_arrays):
    di, dj = s
    def f(x):
        if isinstance(x, Shift) and isinstance(x.a, Var):
            a, b = x.di + di, x.dj + dj
            return x.a if a == b == 0 else Shift(x.a, a, b)
        if isinstance(x, Var) and x.name in grid_arrays:
            return Shift(x, di, dj) if (di, dj) != (0, 0) else x
        if isinstance(x, Idx):
            d = di if x.axis == 'i' else dj
            return Bin('+', x, Lit(float(d))) if d else x
        return None
    return rebuild(e, f)


# ----------------------------------------------------------------- z3 helpers
class Z:
    def __init__(self, int_names=('NX', 'NY'), rename=None):
        self.v, self.ints, self.rename = {}, set(int_names), rename or {}

    def var(self, name, real=True):
        name = self.rename.get(name, name)
        if name not in self.v:
            self.v[name] = z3.Int(name) if name in self.ints or not real else z3.Real(name)
        return self.v[name]

    def tr(self, e):
        t = type(e)
        if t is Lit:
            return z3.RealVal(repr(float(e.v)))
        if t is Idx: return self.var(e.axis, real=False)
        if t is Var: return self.var(e.name, real=e.name not in self.ints)
        if t is Shift:
            return z3.Real(f'{e.a.name}@{e.di:+d},{e.dj:+d}') if (e.di, e.dj) != (0, 0) else self.var(e.a.name)
        if t is Bin:
            a, b = self.tr(e.a), self.tr(e.b)
            a, b = self._coerce(a, b)
            return {'+': a + b, '-': a - b, '*': a * b, '/': a / b}[e.op]
        if t is Un:
            a = self.tr(e.a)
            if e.op == 'neg': return -a
            if e.op == 'abs': return z3.If(a >= 0, a, -a)
            raise ValueError(f'z3: {e.op}')
        if t is Cmp:
            a, b = self._coerce(self.tr(e.a), self.tr(e.b))
            return {'==': a == b, '!=': a != b, '<': a < b, '<=': a <= b, '>': a > b, '>=': a >= b}[e.op]
        if t is BoolOp:
            if e.op == 'not': return z3.Not(self.tr(e.a))
            return (z3.And if e.op == 'and' else z3.Or)(self.tr(e.a), self.tr(e.b))
        if t is Where:
            a, b = self._coerce(self.tr(e.a), self.tr(e.b))
            return z3.If(self.tr(e.c), a, b)
        raise ValueError(f'z3: {t.__name__}')

    @staticmethod
    def _coerce(a, b):
        if z3.is_int(a) and z3.is_real(b): a = z3.ToReal(a)
        if z3.is_real(a) and z3.is_int(b): b = z3.ToReal(b)
        return a, b

    def grid(self, di=0, dj=0):
        i, j, NX, NY = self.var('i', False), self.var('j', False), self.var('NX', False), self.var('NY', False)
        return z3.And(i + di >= 1, i + di <= NX, j + dj >= 1, j + dj <= NY, NX >= 1, NY >= 1)

    def guard(self, g):
        return z3.And([self.tr(c) if pol else z3.Not(self.tr(c)) for c, pol in g]) if g else z3.BoolVal(True)

    def valid(self, hyp, goal):
        s = z3.Solver(); s.set('timeout', 20000)
        s.add(hyp); s.add(z3.Not(goal))
        return s.check() == z3.unsat


def group_terms(terms):
    by = {}
    for s, g, c in terms:
        by.setdefault(s, {}).setdefault(ckey(c), [c, []])[1].append(g)
    return by     # shift -> ckey -> [coef, [guards...]]  (disjunction of guards)


# ------------------------------------------------------------------- symmetry
def symmetry(amul, x='X', grid_arrays=('CK', 'DG')):
    e = inline(amul)
    out = next(iter(e.values()))
    terms = linear_terms(out, x)
    by = group_terms(terms)
    Zc = Z()
    res = {'shifts': sorted(by), 'pairs': [], 'symmetric': True}
    for s, coefs in by.items():
        if s == (0, 0) or s < (0, 0): continue
        ms = (-s[0], -s[1])
        if ms not in by: res['symmetric'] = False; res['pairs'].append({'shift': s, 'ok': False}); continue
        for ck_, (c, gs) in coefs.items():
            match = None
            for ck2, (c2, gs2) in by[ms].items():
                if ckey(shift_expr(c2, s, grid_arrays)) == ck_:
                    match = (c2, gs2)
            ok = match is not None
            if ok:
                G = z3.Or([Zc.guard(g) for g in gs])
                Gs = z3.Or([Zc.guard([(shift_expr(cc, s, grid_arrays), pol) for cc, pol in g]) for g in match[1]])
                ok = Zc.valid(z3.And(Zc.grid(), Zc.grid(*s)), G == Gs) and Zc.valid(Zc.grid(), z3.Implies(G, Zc.grid(*s)))
            res['pairs'].append({'shift': s, 'coef': ir.pp(c), 'partner': ir.pp(match[0]) if match else None,
                                 'ok': bool(ok)})
            res['symmetric'] &= bool(ok)
    diag = by.get((0, 0), {})
    res['diagonal'] = [ir.pp(c) for c, _ in diag.values()]
    res['diag_unconditional'] = len(diag) == 1 and Zc.valid(Zc.grid(), z3.Or([Zc.guard(g) for g in next(iter(diag.values()))[1]]))
    res['terms'] = by
    return res


def jacobi(sym, pcond, x='R'):
    """preconditioner z = x / d  with d the (unconditional) diagonal of A."""
    e = inline(pcond); out = next(iter(e.values()))
    if not (sym['diag_unconditional'] and isinstance(out, Bin) and out.op == '/'
            and key(out.a) == x and len(sym['diagonal']) == 1):
        return False, None
    d = next(iter(sym['terms'][(0, 0)].values()))[0]
    return key(out.b) == key(d), ir.pp(d)


# ------------------------------------------------------------------- intervals
def interval(e, env):
    t = type(e)
    if t is Lit: return (e.v, e.v)
    if t in (Var, Idx):
        return env[e.name if t is Var else e.axis]
    if t is Shift: return interval(e.a, env)
    if t is Un:
        lo, hi = interval(e.a, env)
        if e.op == 'neg': return (-hi, -lo)
        if e.op in ('sin', 'cos'): return (-1.0, 1.0)
        if e.op == 'atan': return (math.atan(lo), math.atan(hi))
        if e.op == 'sqrt': return (math.sqrt(max(lo, 0)), math.sqrt(hi))
        if e.op == 'abs': return (0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi)), max(abs(lo), abs(hi)))
    if t is Bin:
        a, b = interval(e.a, env), interval(e.b, env)
        if e.op == '+': return (a[0] + b[0], a[1] + b[1])
        if e.op == '-': return (a[0] - b[1], a[1] - b[0])
        if e.op == '*':
            ps = [_m(x, y) for x in a for y in b]; return (min(ps), max(ps))
        if e.op == '/':
            if b[0] <= 0 <= b[1]: return (-math.inf, math.inf)
            inv = (1 / b[1], 1 / b[0]) if b[1] != math.inf else (0.0, 1 / b[0])
            ps = [_m(x, y) for x in a for y in inv]; return (min(ps), max(ps))
    if t is Where:
        a, b = interval(e.a, env), interval(e.b, env)
        return (min(a[0], b[0]), max(a[1], b[1]))
    raise ValueError(f'interval: {t.__name__}')


def _m(x, y):
    if (x == 0 and math.isinf(y)) or (y == 0 and math.isinf(x)): return 0.0
    return x * y


# ------------------------------------------------------------------------ SPD
def spd(sym, setup, rename={'NXI': 'NX', 'NYI': 'NY'}):
    """Diagonal dominance & positivity of the operator whose coefficient fields
    are produced by SETUP, at a generic grid point, for all grid sizes (over R)."""
    so_full = inline(setup)
    so = inline(setup, abstract_outputs=True)
    rn = lambda e: rebuild(e, lambda x: Var(rename[x.name]) if isinstance(x, Var) and x.name in rename else None)
    so = {k: rn(v) for k, v in so.items()}
    so_full = {k: rn(v) for k, v in so_full.items()}
    out = {}
    big = (1.0, math.inf)
    iv = interval(so_full['CK'], {'i': big, 'j': big, 'NX': big, 'NY': big})
    Zc = Z()
    NX, NY = Zc.var('NX', False), Zc.var('NY', False)
    # scalar mesh factors: z3 (nonlinear real arithmetic) over all NX, NY >= 1
    hxz, hyz = Zc.tr(so_full['HX2I']), Zc.tr(so_full['HY2I'])
    hx_pos = Zc.valid(z3.And(NX >= 1, NY >= 1), hxz > 0)
    hy_pos = Zc.valid(z3.And(NX >= 1, NY >= 1), hyz > 0)
    out['CK_range'] = list(iv); out['HX2I>0'] = hx_pos; out['HY2I>0'] = hy_pos
    pos = iv[0] > 0 and hx_pos and hy_pos
    hyp = [Zc.grid()]
    for s in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:          # CK at the stencil points
        c = Zc.tr(Shift(Var('CK'), *s)) if s != (0, 0) else Zc.var('CK')
        hyp += [c >= z3.RealVal(str(iv[0])), c <= z3.RealVal(str(iv[1]))]
    hyp += [Zc.var('HX2I') > 0, Zc.var('HY2I') > 0]
    diag = Zc.tr(so['DG'])
    off = []
    for s, coefs in sym['terms'].items():
        if s == (0, 0): continue
        for c, gs in coefs.values():
            g = z3.Or([Zc.guard(x) for x in gs])
            cz = Zc.tr(c)
            off.append(z3.If(g, z3.If(cz >= 0, cz, -cz), 0))
    margin = diag - z3.Sum(off)
    out['diag_positive'] = pos and Zc.valid(z3.And(hyp), diag > 0)
    out['dominant_everywhere'] = pos and Zc.valid(z3.And(hyp), margin >= 0)
    i = Zc.var('i', False)
    out['strict_on_boundary'] = pos and Zc.valid(z3.And(hyp + [i == 1]), margin > 0)
    out['offdiag_nonpositive'] = all(
        Zc.valid(z3.And(hyp), Zc.tr(c) <= 0) for s, cs in sym['terms'].items() if s != (0, 0) for c, _ in cs.values())
    out['SPD'] = bool(sym['symmetric'] and out['diag_positive'] and out['dominant_everywhere']
                      and out['strict_on_boundary'])
    return out


# --------------------------------------------------------------------- degree
TOP = None


def degree(defn, wrt, arrays=()):
    """polynomial degree of every output in parameter `wrt` (None = not polynomial)."""
    outs = inline(defn)
    collapsed = []

    def d(e):
        t = type(e)
        if t in (Var,): return 1 if e.name == wrt else 0
        if t is Shift: return d(e.a)
        if t in (Lit, Idx): return 0
        if t is Bin:
            a, b = d(e.a), d(e.b)
            if a is None or b is None: return None
            if e.op in '+-': return max(a, b)
            if e.op == '*': return a + b
            if e.op == '/': return a if b == 0 else None
        if t is Un:
            a = d(e.a)
            if e.op == 'neg': return a
            return 0 if a == 0 else None
        if t is Where:
            dc = d_cond(e.c)
            a, b = d(e.a), d(e.b)
            if dc == 0:
                return None if a is None or b is None else max(a, b)
            if r_collapse(e):
                collapsed.append(ir.pp(e.c)); return b
            return None
        if t is Fold: return d(e.body)
        if t is ir.Elem: return d(e.a)
        return None

    def d_cond(c):
        return 0 if not any(isinstance(n, Var) and n.name == wrt for n in nodes(c)) else 1

    def r_collapse(w):
        """Where(p == k, a, b): is b[p := k] == a over R ?  (then the branch only matters in IEEE)"""
        c = w.c
        if not (isinstance(c, Cmp) and c.op == '==' and isinstance(c.a, Var) and isinstance(c.b, Lit)):
            return False
        bsub = rebuild(w.b, lambda x: c.b if isinstance(x, Var) and x.name == c.a.name else None)
        try:
            Zc = Z(int_names=())
            return Zc.valid(z3.BoolVal(True), Zc.tr(bsub) == Zc.tr(w.a))
        except ValueError:
            return False

    res = {k: d(v) for k, v in outs.items()}
    return res, collapsed
