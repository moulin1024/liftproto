"""Static semantic lifter: F77 unit -> functional IR term + proof obligations.

Two modes, chosen automatically:
  kernel        : loops are summarised as map / fold / grid-stencil combinators;
                  top-level forward GOTOs and RETURNs are explored path by path
                  (z3 prunes infeasible paths) and the paths are merged again.
  orchestration : straight-line code with CALLs and a backward-GOTO loop becomes
                  a let-sequence with a while_loop (loop rotation); hand-partitioned
                  work arrays are split into named segments (partition proved by z3).
Anything outside the supported fragment raises Opaque(reason) -> work for the LLM.
"""
import copy
from dataclasses import dataclass, field
import z3

from .. import ir
from ..ir import Var, Lit, Bin, Un, Shift, Idx, Cmp, BoolOp, Where, Fold, App, Concat, Let, While, Def
from .fsyntax import Num, Name, Ref, BinX, NegX, RelX, LogX, St, Loop, If
from .poly import Poly


class Opaque(Exception):
    pass


# ---------------------------------------------------------------- symbolic nodes
@dataclass(frozen=True, eq=False)
class ARef(ir.E):            # array element read inside a loop body
    name: str; idx: Poly
@dataclass(frozen=True, eq=False)
class Carried(ir.E):         # value of a scalar at loop-iteration entry
    name: str
@dataclass(frozen=True, eq=False)
class IVal(ir.E):            # integer value used in floating-point context
    p: Poly
@dataclass(frozen=True, eq=False)
class RedAcc(ir.E):          # pending sequential-sum accumulator: init + segments
    init: ir.E; segs: tuple


@dataclass
class Seg:                   # a loop's contribution to an array or an accumulator
    lo: Poly; hi: Poly; f: ir.E; grid: tuple = None   # grid = (NXpoly, NYpoly) for 2-D maps


def key(e):
    """Canonical string of a (symbolic) expression; identity of evaluation order."""
    t = type(e)
    if t is ARef: return f'{e.name}({e.idx})'
    if t is Carried: return f'<{e.name}>'
    if t is IVal: return f'int({e.p})'
    if t is RedAcc: return f'red({key(e.init)};' + ';'.join(f'{s.lo}..{s.hi}:{key(s.f)}' for s in e.segs) + ')'
    if t is Var: return e.name
    if t is Lit: return repr(float(e.v))
    if t is Bin: return f'({key(e.a)}{e.op}{key(e.b)})'
    if t is Un: return f'{e.op}({key(e.a)})'
    if t is Shift: return f'{key(e.a)}[{e.di},{e.dj}]'
    if t is Idx: return e.axis
    if t is Cmp: return f'({key(e.a)}{e.op}{key(e.b)})'
    if t is BoolOp: return f'{e.op}({key(e.a)},{key(e.b) if e.b is not None else ""})'
    if t is Where: return f'ite({key(e.c)},{key(e.a)},{key(e.b)})'
    if t is Fold: return f'fold({key(e.body)})'
    if t is App: return f'{e.fn}(' + ','.join(key(a) for a in e.args) + ')'
    if t is Concat: return '[' + ','.join(key(a) for a in e.parts) + ']'
    if t is ir.Elem: return f'{key(e.a)}({key(e.i)})'
    raise TypeError(t)


def rebuild(e, fn, memo=None):
    """Bottom-up rewrite: fn(node) -> replacement or None.  Shared sub-terms stay shared."""
    memo = {} if memo is None else memo
    if id(e) in memo: return memo[id(e)][1]
    r = fn(e)
    if r is None:
        ch = _children(e)
        if ch:
            new = [rebuild(c, fn, memo) for c in ch]
            r = e if all(a is b for a, b in zip(new, ch)) else rebuild_children(e, new)
        else:
            r = e
    memo[id(e)] = (e, r)
    return r


def nodes(e):
    yield e
    for c in _children(e):
        yield from nodes(c)


# ------------------------------------------------------------------- conditions
# integer conditions are trees ('rel', op, Poly, Poly) | ('and'|'or', c1, c2) | ('not', c)
REL = {'EQ': '==', 'NE': '!=', 'LT': '<', 'LE': '<=', 'GT': '>', 'GE': '>='}


def icond_z3(c, env):
    if c[0] == 'rel':
        a, b = c[2].z3(env), c[3].z3(env)
        return {'==': a == b, '!=': a != b, '<': a < b, '<=': a <= b, '>': a > b, '>=': a >= b}[c[1]]
    if c[0] == 'not': return z3.Not(icond_z3(c[1], env))
    f = z3.And if c[0] == 'and' else z3.Or
    return f(icond_z3(c[1], env), icond_z3(c[2], env))


def icond_str(c):
    if c[0] == 'rel': return f'{c[2]} {c[1]} {c[3]}'
    if c[0] == 'not': return f'not ({icond_str(c[1])})'
    return f'({icond_str(c[1])} {c[0]} {icond_str(c[2])})'


def neg(c): return ('not', c)


# -------------------------------------------------------------------- the context
@dataclass
class Summary:               # what callers need to know about a lifted unit
    name: str
    kind: str                # 'kernel' | 'orchestration' | 'opaque'
    term: Def = None
    params: list = field(default_factory=list)      # non-integer formals passed to App, in order
    outs: list = field(default_factory=list)        # written formals ('__result__' for functions)
    footprint: dict = field(default_factory=dict)   # array formal / COMMON array -> hi Poly (lo = 1)
    int_formals: list = field(default_factory=list)
    reads_common: list = field(default_factory=list)
    writes_common: list = field(default_factory=list)
    reason: str = None


class Prover:
    """Collects symbol definitions/assumptions and discharges obligations with z3."""

    def __init__(self):
        self.defs, self.env, self.assumptions, self.log = [], {}, [], []

    def fresh_mod(self, a, b, tag):
        s = f'{tag}'
        self.defs.append((s, a, b))
        return Poly.sym(s)

    def _base(self, sol):
        for s, a, b in self.defs:
            av, bv, sv = a.z3(self.env), b.z3(self.env), Poly.sym(s).z3(self.env)
            # Fortran MOD truncates toward zero
            sol.add(sv == z3.If(av >= 0, av % bv, -((-av) % bv)))
        for c in self.assumptions:
            sol.add(icond_z3(c, self.env))

    def sat(self, conds):
        s = z3.Solver(); s.set('timeout', 5000); self._base(s)
        for c in conds: s.add(icond_z3(c, self.env) if isinstance(c, tuple) else c)
        r = s.check()
        return r != z3.unsat

    def valid(self, conds, goal):
        """conds => goal ?  (goal: icond tree or z3 expr)"""
        g = icond_z3(goal, self.env) if isinstance(goal, tuple) else goal
        return not self.sat(list(conds) + [z3.Not(g)])

    def counterexample(self, conds, goal):
        g = icond_z3(goal, self.env) if isinstance(goal, tuple) else goal
        s = z3.Solver(); s.set('timeout', 5000); self._base(s)
        for c in conds: s.add(icond_z3(c, self.env) if isinstance(c, tuple) else c)
        s.add(z3.Not(g))
        if s.check() != z3.sat: return None
        m = s.model()
        def syms(c):
            if isinstance(c, tuple):
                if c[0] == 'rel': return c[2].syms() | c[3].syms()
                return set().union(*(syms(x) for x in c[1:]))
            return set()
        rel = syms(goal).union(*(syms(c) for c in conds)) if conds else syms(goal)
        return ', '.join(f'{d}={m[d]}' for d in sorted(m.decls(), key=str)
                         if '_mod' not in str(d) and str(d) in rel)

    def record(self, unit, kind, claim, ok, line=None, detail=None):
        self.log.append({'unit': unit, 'kind': kind, 'claim': claim, 'status': 'proved' if ok else 'FAILED',
                         'line': line, 'detail': detail})
        return ok


# ------------------------------------------------------------------ unit lifter
class State:
    def __init__(self):
        self.ints, self.dbl, self.arr = {}, {}, {}   # arr: name -> (value E, [Seg])
        self.pc, self.pcd = [], []                    # integer path conditions, (cond E, polarity)
        self.ret = False
        self.elem = {}                                # loop-body element writes: (name, key) -> E

    def copy(self):
        s = State()
        s.ints, s.dbl = dict(self.ints), dict(self.dbl)
        s.arr = {k: (v, list(g)) for k, (v, g) in self.arr.items()}
        s.pc, s.pcd, s.ret, s.elem = list(self.pc), list(self.pcd), self.ret, dict(self.elem)
        return s


INTR_D = {'DSQRT': 'sqrt', 'SQRT': 'sqrt', 'DSIN': 'sin', 'SIN': 'sin', 'DCOS': 'cos', 'COS': 'cos',
          'DATAN': 'atan', 'ATAN': 'atan', 'ABS': 'abs', 'DABS': 'abs'}


class UnitLifter:
    def __init__(self, unit, body, units, summaries, prover):
        self.u, self.body, self.units, self.S, self.P = unit, body, units, summaries, prover
        self.commons = {n: blk for blk, names in unit.commons.items() for n in names}
        self.formals = list(unit.args)
        self.reads, self.writes = set(), set()          # formals / commons touched
        self.footprint = {}                             # array -> list of hi Poly
        self.lets, self.ncount = [], 0
        self.loopctx = None                             # set while executing a loop body
        self.branch_seen = {}                           # (line, side) -> feasible on some path
        self.loop_ztrip = {}                            # line -> zero-trip feasible on some path
        self.sections = {}                              # orchestration: array -> {offset Poly: segname}
        self.mode = 'kernel'
        self.call_sites, self.contracts, self.unknown_fp = [], [], []
        self.read_extent, self.name_lens = {}, {}
        self.rt_params, self.extra_fp = set(), {}
        self.specialized = {}
        self._last_cov = {}

    # ---------------------------------------------------------------- helpers
    def is_int(self, n):
        if self.loopctx and n in self.loopctx['vars']: return True
        return self.u.var(n).type == 'INTEGER'

    def is_array(self, n):
        return bool(self.u.dims.get(n))

    def note_read(self, n):
        if n in self.formals or n in self.commons: self.reads.add(n)

    def note_write(self, n):
        if n in self.formals or n in self.commons: self.writes.add(n)

    def lit(self, v): return Lit(float(v))

    def iv2ir(self, p):
        """integer polynomial -> IR (inside grid loops loop vars become Idx)."""
        if p.is_const(): return Lit(float(p.cval()))
        lc = self.loopctx
        out = None
        for mono, c in sorted(p.t.items()):
            if not mono:
                term = Lit(float(c))
            else:
                fs = []
                for s in mono:
                    if lc and lc.get('grid') and s == lc['grid'][0]: fs.append(Idx('i'))
                    elif lc and lc.get('grid') and s == lc['grid'][1]: fs.append(Idx('j'))
                    elif lc and s in lc['vars']: raise Opaque(f'loop index {s} used as a value in a 1-D loop')
                    else: fs.append(Var(s))
                term = fs[0]
                for f in fs[1:]: term = Bin('*', term, f)
                if c != 1: term = Bin('*', Lit(float(c)), term)
            out = term if out is None else Bin('+', out, term)
        return out

    def dval(self, x):
        """coerce an evaluated value to floating-point IR."""
        k, v = x
        if k == 'dbl': return v
        if k == 'int':
            if v.is_const(): return Lit(float(v.cval()))
            return IVal(v) if self.loopctx is not None else self.iv2ir(v)
        raise Opaque('condition used as value')

    # ------------------------------------------------------------- evaluation
    def ev(self, e, st):
        t = type(e)
        if t is Num:
            return ('int', Poly.const(e.v)) if e.isint else ('dbl', Lit(e.v))
        if t is Name:
            n = e.n
            if n in self.u.params:
                v = self.u.params[n]
                return ('int', Poly.const(v)) if isinstance(v, int) else ('dbl', Lit(float(v)))
            if self.is_int(n):
                if n in self.specialized and n not in st.ints:
                    return ('int', Poly.const(self.specialized[n]))
                if n in st.ints:
                    v = st.ints[n]
                    if v == 'CARRIED': raise Opaque(f'loop-carried integer {n}')
                    if isinstance(v, ir.E): return ('dbl', v)          # runtime integer (orchestration)
                    return ('int', v)
                self.note_read(n)
                return ('int', Poly.sym(n))
            if self.is_array(n):
                raise Opaque(f'whole-array reference {n} in expression')
            if n in st.dbl:
                v = st.dbl[n]
                if isinstance(v, Carried) and v.name.startswith('__undef__'):
                    raise Opaque(f'{n} may be used undefined')
                return ('dbl', v)
            self.note_read(n)
            return ('dbl', Var(n))
        if t is Ref:
            n = e.n
            if n == 'MOD':
                a, b = (self.ev(x, st) for x in e.args)
                if a[0] != 'int' or b[0] != 'int' or not b[1].is_const(): raise Opaque('MOD of non-integers')
                self.ncount += 1
                return ('int', self.P.fresh_mod(a[1], b[1], f'{self.u.name}_mod{self.ncount}'))
            if n == 'DBLE':
                return ('dbl', self.dval(self.ev(e.args[0], st)))
            if n in INTR_D:
                return ('dbl', Un(INTR_D[n], self.dval(self.ev(e.args[0], st))))
            if self.is_array(n):
                idx = self.ev(e.args[0], st)
                if len(e.args) != 1 or idx[0] != 'int': raise Opaque(f'non-integer or multi-dim index into {n}')
                self.note_read(n)
                return ('dbl', self.read_elem(n, idx[1], st))
            if n in self.units or n in self.S:                       # user function call
                return ('dbl', self.call_expr(n, e.args, st, is_function=True))
            raise Opaque(f'unknown function {n}')
        if t is BinX:
            a, b = self.ev(e.a, st), self.ev(e.b, st)
            if a[0] == 'int' and b[0] == 'int' and e.op in '+-*':
                return ('int', {'+': a[1] + b[1], '-': a[1] - b[1], '*': a[1] * b[1]}[e.op])
            if e.op == '**':
                if b[0] == 'int' and b[1] == 2:
                    x = self.dval(a); return ('dbl', Bin('*', x, x))
                raise Opaque('general exponentiation')
            if a[0] == 'int' and b[0] == 'int':
                raise Opaque('integer division')
            return ('dbl', Bin(e.op, self.dval(a), self.dval(b)))
        if t is NegX:
            a = self.ev(e.a, st)
            return ('int', -a[1]) if a[0] == 'int' else ('dbl', Un('neg', self.dval(a)))
        if t is RelX:
            a, b = self.ev(e.a, st), self.ev(e.b, st)
            if a[0] == 'int' and b[0] == 'int':
                return ('icond', ('rel', REL[e.op], a[1], b[1]))
            return ('dcond', Cmp(REL[e.op], self.dval(a), self.dval(b)))
        if t is LogX:
            a = self.ev(e.a, st)
            if e.op == 'NOT':
                return ('icond', neg(a[1])) if a[0] == 'icond' else ('dcond', BoolOp('not', self.c2ir(a)))
            b = self.ev(e.b, st)
            if a[0] == 'icond' and b[0] == 'icond':
                return ('icond', (e.op.lower(), a[1], b[1]))
            return ('dcond', BoolOp(e.op.lower(), self.c2ir(a), self.c2ir(b)))
        raise Opaque(f'unsupported expression {t.__name__}')

    def c2ir(self, c):
        k, v = c
        if k == 'dcond': return v
        def tr(t):
            if t[0] == 'rel': return Cmp(t[1], self.iv2ir(t[2]), self.iv2ir(t[3]))
            if t[0] == 'not': return BoolOp('not', tr(t[1]))
            return BoolOp(t[0], tr(t[1]), tr(t[2]))
        return tr(v)

    def read_elem(self, n, idx, st):
        if self.mode == 'orchestration':
            n, idx = self.resolve_section(n, idx)
        if self.loopctx is None:
            if self.mode == 'kernel':
                self.read_extent.setdefault(n, []).append(idx)
                return ir.Elem(Var(n), self.iv2ir(idx))
            raise Opaque(f'array element {n}({idx}) outside a loop')
        if (n, str(idx)) in st.elem:
            return st.elem[(n, str(idx))]
        v, segs = self.array_value(n, st)
        lc = self.loopctx
        if segs and lc.get('grid') is None and all(
                self.P.valid(st.pc, ('rel', '<', sg.hi, lc['lo'])) for sg in segs):
            pass                                  # earlier loops wrote only elements below this loop's range
        else:
            self.normalize_array(n, st)
        self.loopctx['reads'].append((n, idx, list(self.loopctx['conds'])))
        return ARef(n, idx)

    # ---------------------------------------------------------- normalization
    def array_value(self, n, st):
        return st.arr.get(n, (Var(n), []))

    def normalize_array(self, n, st, final=False):
        v, segs = self.array_value(n, st)
        if not segs: return v
        f = self.merge_segments(n, segs, st)
        if self.mode == 'kernel' and not final:
            if not any(isinstance(l, Let) and l.name == n and key(l.e) == key(f) for l in self.lets):
                self.lets.append(Let(n, f))
            st.arr[n] = (Var(n), [])
            return Var(n)
        st.arr[n] = (f, [])
        return f

    def merge_segments(self, what, segs, st, init=None):
        k0 = key(segs[0].f)
        if any(key(s.f) != k0 for s in segs):
            raise Opaque(f'{what}: loops write different functions to different ranges')
        if any(s.grid != segs[0].grid for s in segs):
            raise Opaque(f'{what}: mixed grid/1-D segments')
        ok = self.P.valid(st.pc, ('rel', '==', segs[0].lo, Poly.const(1)))
        for a, b in zip(segs, segs[1:]):
            ok &= self.P.valid(st.pc, ('rel', '==', b.lo, a.hi + 1))
        self.P.record(self.u.name, 'coverage', f'{what}: ' + ' ++ '.join(f'[{s.lo}..{s.hi}]' for s in segs)
                      + f' = [1..{segs[-1].hi}]', ok, detail=' and '.join(icond_str(c) for c in st.pc) or 'true')
        if not ok:
            goal = ('rel', '==', segs[0].lo, Poly.const(1))
            for x, y in zip(segs, segs[1:]):
                goal = ('and', goal, ('rel', '==', y.lo, x.hi + 1))
            cex = self.P.counterexample(st.pc, goal)
            raise Opaque(f'{what}: loop segments do not tile [1..n]'
                         + (f' (counterexample: {cex})' if cex else ' (could not prove)'))
        self.footprint.setdefault(what, []).append(segs[-1].hi)
        if not hasattr(self, '_last_cov'): self._last_cov = {}
        self._last_cov[what] = segs[-1].hi
        return segs[0].f

    def norm_scalar(self, v, st):
        if isinstance(v, RedAcc):
            if key(v.init) != key(Lit(0.0)):
                raise Opaque('reduction with non-zero initial value')
            if not v.segs:
                return Lit(0.0)
            body = self.merge_segments('reduction', list(v.segs), st)
            return Fold(body)
        return v

    # -------------------------------------------------------------- top level
    def lift(self):
        kinds = self._scan(self.body)
        if 'call' in kinds or 'backgoto' in kinds or self._has_fn_call():
            self.mode = 'orchestration'
            return self.lift_orchestration()
        return self.lift_kernel()

    def _scan(self, stmts, found=None, labels=None):
        found = set() if found is None else found
        labels = {} if labels is None else labels
        for i, s in enumerate(stmts):
            if isinstance(s, Loop): self._scan(s.body, found); continue
            if isinstance(s, If):
                for _, b in s.branches: self._scan(b, found)
                self._scan(s.els, found); continue
            if s.label is not None: labels[s.label] = i
            if s.kind == 'call': found.add('call')
            if s.kind == 'goto' and s.a['target'] in labels: found.add('backgoto')
        return found

    def _has_fn_call(self):
        def walk(x):
            if isinstance(x, Ref) and x.n in self.S and not self.is_array(x.n): return True
            if isinstance(x, (BinX, RelX, LogX)): return walk(x.a) or (x.b is not None and walk(x.b))
            if isinstance(x, NegX): return walk(x.a)
            if isinstance(x, Ref): return any(walk(a) for a in x.args)
            return False
        def sts(stmts):
            for s in stmts:
                if isinstance(s, Loop):
                    if sts(s.body): return True
                elif isinstance(s, If):
                    if any(walk(c) or sts(b) for c, b in s.branches) or sts(s.els): return True
                elif s.kind == 'assign' and walk(s.a['rhs']): return True
            return False
        return sts(self.body)

    # ================================================================ KERNEL MODE
    def lift_kernel(self):
        paths = list(self.exec_top(self.body, 0, State()))
        u = self.u
        outs_order = [a for a in self.formals if a in self.writes and not self.is_int(a)]
        if u.kind == 'FUNCTION': outs_order = ['__result__'] + outs_order
        outs_order += [c for c in self.commons if c in self.writes]
        results = {}
        for o in outs_order:
            name = u.name if o == '__result__' else o
            vals = []
            for st in paths:
                if self.is_array(name):
                    had = bool(self.array_value(name, st)[1])
                    v = self.normalize_array(name, st, final=True)
                    cov = self.footprint.get(name, [None])[-1] if had else self._last_cov.get(name)
                else:
                    if self.is_int(name):
                        x = st.ints.get(name, Poly.sym(name))
                        v = self.iv2ir(x) if isinstance(x, Poly) else x
                        cov = None
                    else:
                        raw = st.dbl.get(name, Var(name))
                        cov = raw.segs[-1].hi if isinstance(raw, RedAcc) and raw.segs else None
                        v = self.norm_scalar(raw, st)
                        if isinstance(v, Fold): self.footprint.setdefault('reduction', []).append(cov)
                vals.append((st, v, cov))
            results[o] = self.merge_paths(o, vals)
        params = [a for a in self.formals if a in self.reads and not self.is_int(a)]
        intf = [a for a in self.formals if self.is_int(a)]
        term = Def(u.name, params, list(self.lets), {o: e for o, e in results.items()})
        term = cse(term)
        used = {x.name for e in list(term.outs.values()) + [l.e for l in term.body] for x in nodes(e)
                if isinstance(x, Var)}
        params = [a for a in self.formals if a in params or (self.is_int(a) and a in used)]
        term = Def(term.name, params, term.body, term.outs)
        fp = self.footprints()
        return Summary(u.name, 'kernel', term, params, outs_order, fp, intf,
                       sorted(c for c in self.commons if c in self.reads),
                       sorted(c for c in self.commons if c in self.writes))

    def footprints(self):
        fresh = {d[0] for d in self.P.defs}
        def pick(his):
            his = [h for h in his if h is not None]
            for h in his:
                if not (h.syms() & fresh): return h
            return his[0] if his else None
        fp = {}
        for n, his in list(self.footprint.items()) + list(self.read_extent.items()):
            if n == 'reduction' or n in fp: continue
            h = pick(his if isinstance(his, list) else [his])
            if h is not None: fp[n] = h
        return fp

    def merge_paths(self, what, vals):
        """vals: [(state, value, coverage-hi)] -> one expression."""
        fresh = {d[0] for d in self.P.defs}
        groups = []
        for st, v, cov in vals:
            k = key(v)
            for g in groups:
                if g['key'] == k:
                    g['paths'].append((st, cov))
                    if g['cov'] is None or (cov is not None and (g['cov'].syms() & fresh) and not (cov.syms() & fresh)):
                        g['cov'] = cov
                    break
            else:
                groups.append({'key': k, 'v': v, 'cov': cov, 'paths': [(st, cov)]})
        for g in groups:                      # same expression must also mean the same extent on every path
            if g['cov'] is None: continue
            for st, cov in g['paths']:
                if cov is not None and not self.P.valid(st.pc, ('rel', '==', cov, g['cov'])):
                    raise Opaque(f'{what}: same expression over different extents ({cov} vs {g["cov"]})')
            self.P.record(self.u.name, 'extent', f'{what}: every path produces the result over [1..{g["cov"]}]',
                          True, detail=f'{len(g["paths"])} path(s)')
        # absorb individual paths on which a group's domain is empty (vacuous equality)
        for g in groups:
            if g['cov'] is None: continue
            for h in groups:
                if h is g: continue
                keep = []
                for st, cov in h['paths']:
                    if self.P.valid(st.pc, ('rel', '<', g['cov'], Poly.const(1))):
                        self.P.record(self.u.name, 'vacuous-path',
                                      f'{what}: path [{" and ".join(icond_str(c) for c in st.pc)}] has empty '
                                      f'domain [1..{g["cov"]}] -> agrees with the main result', True)
                        g.setdefault('absorbed', []).append((st, cov))
                    else:
                        keep.append((st, cov))
                h['paths'] = keep
        groups = [g for g in groups if g['paths']]
        if len(groups) == 1:
            return groups[0]['v']
        # remaining groups must be told apart by floating-point (data) conditions
        for g in groups:
            for h in groups:
                if g is h: continue
                for st1, _ in g['paths']:
                    for st2, _ in h['paths']:
                        if not any(key(c1) == key(c2) and p1 != p2 for c1, p1 in st1.pcd for c2, p2 in st2.pcd):
                            raise Opaque(f'{what}: result depends on an integer path condition')
        result = groups[-1]['v']
        for g in reversed(groups[:-1]):
            conds = []
            for st, _ in g['paths']:
                lits = [c if pol else BoolOp('not', c) for c, pol in st.pcd]
                c = lits[0]
                for l in lits[1:]: c = BoolOp('and', c, l)
                conds.append(c)
            c = conds[0]
            for x in conds[1:]: c = BoolOp('or', c, x)
            self.P.record(self.u.name, 'data-dependent-path', f'{what}: separate IEEE semantics when {ir.pp(c)}',
                          True, detail='kept as an explicit Where (not merged)')
            result = Where(c, g['v'], result)
        return result

    def exec_top(self, stmts, i, st):
        labels = {s.label: j for j, s in enumerate(stmts) if not isinstance(s, (Loop, If)) and s.label is not None}
        labels.update({s.label: j for j, s in enumerate(stmts) if isinstance(s, (Loop, If)) and s.label})
        while i < len(stmts):
            s = stmts[i]
            if isinstance(s, Loop):
                self.summarize_loop(s, st); i += 1; continue
            if isinstance(s, If):
                rest = stmts[i + 1:]
                for bst, body in self.fork_if(s, st):
                    yield from self.exec_top(body + rest, 0, bst)
                return
            k = s.kind
            if k in ('continue', 'skip'): i += 1; continue
            if k == 'return' or k == 'end':
                yield st; return
            if k == 'assign':
                self.assign(s, st); i += 1; continue
            if k in ('logif', 'if_goto'):
                c = self.ev(s.a['cond'], st)
                for side, bst in self.fork(c, st, s.line):
                    if side:
                        if k == 'logif':
                            sub = s.a['stmt']
                            if sub.kind == 'return': yield bst; continue
                            if sub.kind == 'goto':
                                yield from self.exec_top(stmts, self._fwd(labels, sub.a['target'], i), bst); continue
                            self.assign(sub, bst)
                            yield from self.exec_top(stmts, i + 1, bst)
                        else:
                            yield from self.exec_top(stmts, self._fwd(labels, s.a['target'], i), bst)
                    else:
                        yield from self.exec_top(stmts, i + 1, bst)
                return
            if k == 'goto':
                i = self._fwd(labels, s.a['target'], i); continue
            raise Opaque(f'line {s.line}: {k} not supported in kernel mode')
        yield st

    def _fwd(self, labels, target, i):
        j = labels.get(target)
        if j is None or j <= i: raise Opaque(f'non-forward GOTO {target}')
        return j

    def fork(self, c, st, line):
        if c[0] == 'icond':
            for side, cc in ((True, c[1]), (False, neg(c[1]))):
                feas = self.P.sat(st.pc + [cc])
                self.branch_seen[(line, side)] = self.branch_seen.get((line, side), False) or feas
                if feas:
                    b = st.copy(); b.pc.append(cc); yield side, b
        else:
            ce = self.c2ir(c)
            for side in (True, False):
                self.branch_seen[(line, side)] = True
                b = st.copy(); b.pcd.append((ce, side)); yield side, b

    def fork_if(self, node, st):
        prev = []
        for cond, body in node.branches:
            c = self.ev(cond, st)
            if c[0] != 'icond': raise Opaque('block IF on floating-point condition at top level')
            b = st.copy(); b.pc += [neg(p) for p in prev] + [c[1]]
            if self.P.sat(b.pc): yield b, body
            prev.append(c[1])
        b = st.copy(); b.pc += [neg(p) for p in prev]
        if self.P.sat(b.pc): yield b, node.els

    def assign(self, s, st):
        lhs, rhs = s.a['lhs'], s.a['rhs']
        val = self.ev(rhs, st)
        if isinstance(lhs, Name):
            n = lhs.n
            self.note_write(n)
            if self.is_int(n):
                if val[0] == 'int': st.ints[n] = val[1]
                else: st.ints[n] = val[1]          # runtime integer (orchestration only)
            else:
                st.dbl[n] = self.dval(val)
            return
        if isinstance(lhs, Ref) and self.is_array(lhs.n):
            idx = self.ev(lhs.args[0], st)
            n, ix = lhs.n, idx[1]
            if self.mode == 'orchestration':
                n, ix = self.resolve_section(n, ix)
            if self.loopctx is None: raise Opaque(f'array element write {lhs.n} outside loop')
            self.note_write(lhs.n)
            v = self.dval(val)
            st.elem[(n, str(ix))] = v
            self.loopctx['writes'].append((n, ix, v, list(self.loopctx['conds'])))
            return
        raise Opaque('unsupported assignment target')

    # ------------------------------------------------------------ loop summary
    def summarize_loop(self, L, st):
        grid = None
        if len(L.body) == 1 and isinstance(L.body[0], Loop):
            inner = L.body[0]
            grid = (inner, L)
        if grid:
            ihi0 = self.ev(grid[0].hi, st)
            if ihi0[0] == 'int':
                self.P.assumptions += [('rel', '>=', ihi0[1], Poly.const(1))]
            ohi0 = self.ev(L.hi, st)
            if ohi0[0] == 'int':
                self.P.assumptions += [('rel', '>=', ohi0[1], Poly.const(1))]
        lo, hi, step = (self.ev(x, st) for x in (L.lo, L.hi, L.step))
        if not (lo[0] == hi[0] == step[0] == 'int') or not step[1].is_const():
            raise Opaque(f'line {L.line}: non-constant loop step or non-integer bounds')
        lo, hi, s = lo[1], hi[1], step[1].cval()
        ztrip = self.P.sat(st.pc + [('rel', '<', hi, lo)])
        self.loop_ztrip[L.line] = self.loop_ztrip.get(L.line, False) or ztrip
        if grid:
            inner, outer = grid
            ilo, ihi, istep = (self.ev(x, st) for x in (inner.lo, inner.hi, inner.step))
            if not (lo == 1 and s == 1 and ilo[1] == 1 and istep[1] == 1):
                raise Opaque('2-D loop nest not starting at 1 with unit stride')
            nxp, nyp = ihi[1], hi
            self.P.assumptions += [('rel', '>=', nxp, Poly.const(1)), ('rel', '>=', nyp, Poly.const(1))]
            self.loop_ztrip[inner.line] = self.loop_ztrip.get(inner.line, False)
            ctx = {'vars': {inner.var, outer.var}, 'grid': (inner.var, outer.var), 'gridpolys': (nxp, nyp),
                   'reads': [], 'writes': [], 'conds': [], 'line': L.line,
                   'range': [('rel', '>=', Poly.sym(inner.var), Poly.const(1)),
                             ('rel', '<=', Poly.sym(inner.var), nxp),
                             ('rel', '>=', Poly.sym(outer.var), Poly.const(1)),
                             ('rel', '<=', Poly.sym(outer.var), nyp)]}
            body = inner.body
        else:
            if s < 1: raise Opaque('non-positive step')
            iv = Poly.sym(L.var)
            ctx = {'vars': {L.var}, 'grid': None, 'reads': [], 'writes': [], 'conds': [], 'line': L.line,
                   'range': [('rel', '>=', iv, lo), ('rel', '<=', iv, hi)],
                   'step': s, 'lo': lo, 'hi': hi}
            body = L.body
        self.loopctx = ctx
        bst = st.copy()
        for n in list(bst.dbl): bst.dbl[n] = Carried(n)
        for n in self._assigned_ints(body): bst.ints[n] = 'CARRIED'
        bst.ints.pop(L.var, None)
        if grid: bst.ints.pop(grid[0].var, None)
        bst.elem = {}
        try:
            bst = self.exec_body(body, bst)
        finally:
            self.loopctx = None
        for n, (v, segs) in bst.arr.items():
            if not segs and self.array_value(n, st)[1]:
                st.arr[n] = (v, [])
        self.loopctx = ctx
        try:
            self._apply_loop(L, st, bst, ctx, grid)
        finally:
            self.loopctx = None

    def _assigned_ints(self, stmts):
        out = set()
        for s in stmts:
            if isinstance(s, Loop): out |= self._assigned_ints(s.body)
            elif isinstance(s, If):
                for _, b in s.branches: out |= self._assigned_ints(b)
                out |= self._assigned_ints(s.els)
            elif s.kind == 'assign' and isinstance(s.a['lhs'], Name) and self.is_int(s.a['lhs'].n):
                out.add(s.a['lhs'].n)
        return out

    def exec_body(self, stmts, st):
        """merging executor for loop bodies (forward GOTOs, logical / block IF)."""
        labels = {s.label: j for j, s in enumerate(stmts) if s.label is not None}
        i = 0
        while i < len(stmts):
            s = stmts[i]
            if isinstance(s, Loop): raise Opaque('loop nest deeper than 2 or not perfectly nested')
            if isinstance(s, If):
                st = self._merge_if(s, st); i += 1; continue
            k = s.kind
            if k in ('continue', 'skip'): i += 1; continue
            if k == 'assign':
                if isinstance(s.a['lhs'], Name) and self.is_int(s.a['lhs'].n):
                    st.ints.pop(s.a['lhs'].n, None)
                self.assign(s, st); i += 1; continue
            if k == 'logif':
                c = self.ev(s.a['cond'], st)
                if s.a['stmt'].kind != 'assign': raise Opaque('logical IF with non-assignment inside loop')
                sub = st.copy()
                self.loopctx['conds'].append(c)
                self.assign(s.a['stmt'], sub)
                self.loopctx['conds'].pop()
                st = self._merge(c, sub, st); i += 1; continue
            if k == 'if_goto':
                j = labels.get(s.a['target'])
                if j is None or j <= i: raise Opaque('non-forward GOTO in loop body')
                c = self.ev(s.a['cond'], st)
                nc = ('icond', neg(c[1])) if c[0] == 'icond' else ('dcond', BoolOp('not', self.c2ir(c)))
                self.loopctx['conds'].append(nc)
                fall = self.exec_body(stmts[i + 1:j], st.copy())
                self.loopctx['conds'].pop()
                st = self._merge(c, st, fall); i = j; continue
            raise Opaque(f'line {s.line}: {k} inside loop body')
        return st

    def _merge_if(self, node, st):
        res, prev = None, []
        branches = list(node.branches) + [(None, node.els)]
        outs = []
        for cond, body in branches:
            c = self.ev(cond, st) if cond is not None else None
            if c is not None: self.loopctx['conds'].append(c)
            outs.append((c, self.exec_body(body, st.copy())))
            if c is not None: self.loopctx['conds'].pop()
        res = outs[-1][1]
        for c, b in reversed(outs[:-1]):
            res = self._merge(c, b, res)
        return res

    def _merge(self, c, a, b):
        """state = c ? a : b"""
        ce = self.c2ir(c)
        m = b.copy()
        for n in set(a.dbl) | set(b.dbl):
            va, vb = a.dbl.get(n), b.dbl.get(n)
            if va is vb: continue
            if va is None or vb is None:
                m.dbl[n] = Carried('__undef__' + n); continue      # only an error if read later
            m.dbl[n] = Where(ce, va, vb)
        for n in set(a.ints) | set(b.ints):
            if not (a.ints.get(n) == b.ints.get(n)): raise Opaque(f'integer {n} differs between branches')
        for kk in set(a.elem) | set(b.elem):
            if a.elem.get(kk) is not b.elem.get(kk): raise Opaque(f'conditional array write {kk[0]}')
        return m

    def _apply_loop(self, L, st, bst, ctx, grid):
        name = self.u.name
        # loop-invariant reads: Carried(v) for v not written in the body
        written = {n for n in bst.dbl if not (isinstance(bst.dbl[n], Carried) and bst.dbl[n].name == n)}
        def inv(e):
            if isinstance(e, Carried) and e.name not in written and not e.name.startswith('__undef__'):
                return st.dbl[e.name]
            return None
        carried = []
        for n in written:
            v = rebuild(bst.dbl[n], inv); bst.dbl[n] = v
            if isinstance(v, Carried) and v.name.startswith('__undef__'):
                continue                                  # body-local temporary
            if any(isinstance(x, Carried) for x in nodes(v)):
                carried.append(n)
        writes = [(a, i, rebuild(v, inv), c) for a, i, v, c in ctx['writes']]
        if carried:
            if writes: raise Opaque(f'loop at line {L.line} mixes reductions and array writes')
            for n in carried:
                self._reduction(n, bst.dbl[n], st, ctx, L)
            # non-carried scalars written in the body are dead after the loop (assumed)
            return
        if not writes:
            return
        if grid:
            self._grid_map(writes, st, ctx, L)
        else:
            self._map1d(writes, st, ctx, L)

    def _reduction(self, n, v, st, ctx, L):
        s = ctx.get('step')
        if s is None: raise Opaque('reduction over a 2-D nest')
        terms = []
        while isinstance(v, Bin) and v.op == '+':
            terms.append(v.b); v = v.a
        if not (isinstance(v, Carried) and v.name == n):
            raise Opaque(f'line {L.line}: loop-carried state ({", ".join(sorted(set(x.name for x in nodes(v) if isinstance(x, Carried)) | {n}))})'
                         ' with an update that is not a sequential sum -> not a recognised reduction')
        terms.reverse()
        if len(terms) != s:
            raise Opaque(f'reduction body adds {len(terms)} terms but loop step is {s}')
        iv = list(ctx['vars'])[0]
        t0 = terms[0]
        for k, tk in enumerate(terms):
            if key(shift_idx(t0, iv, k)) != key(tk):
                raise Opaque(f'unrolled term {k} is not term 0 shifted by {k}')
            if any(isinstance(x, Carried) for x in nodes(tk)): raise Opaque('accumulator used inside summand')
        f = self._elementwise(t0, Poly.sym(iv), ctx)
        hi = self._exact_hi(ctx, st, L)
        for rn, _, _ in ctx['reads']:
            self.read_extent.setdefault(rn, []).append(hi)
        acc = st.dbl.get(n, Var(n))
        seg = Seg(ctx['lo'], hi, f)
        st.dbl[n] = RedAcc(acc.init, acc.segs + (seg,)) if isinstance(acc, RedAcc) else RedAcc(acc, (seg,))

    def _exact_hi(self, ctx, st, L):
        s, lo, hi = ctx['step'], ctx['lo'], ctx['hi']
        if s == 1: return hi
        # last element touched = last iterate + s - 1; equals hi iff (hi - lo + 1) mod s == 0
        env = self.P.env
        e = (hi - lo + 1).z3(env)
        goal = z3.Or(e <= 0, e % s == 0)
        ok = self.P.valid(st.pc, goal)
        self.P.record(self.u.name, 'unroll-exact', f'loop line {L.line}: ({hi} - {lo} + 1) mod {s} == 0 '
                      '(unrolled body covers exactly [lo..hi])', ok,
                      line=L.line, detail=' and '.join(icond_str(c) for c in st.pc) or 'true')
        if not ok: raise Opaque(f'line {L.line}: cannot prove unrolled loop covers exactly [lo..hi]')
        return hi

    def _elementwise(self, e, idxp, ctx, shift_ok=False):
        def tr(x):
            if isinstance(x, ARef):
                d = x.idx - idxp
                if d == 0: return Var(x.name)
                raise Opaque(f'{x.name}: read at offset {d} relative to the element being produced')
            if isinstance(x, IVal):
                return self.iv2ir(x.p)
            if isinstance(x, Carried): raise Opaque('unexpected carried value')
            return None
        return rebuild(e, tr)

    def _map1d(self, writes, st, ctx, L):
        s, iv = ctx['step'], Poly.sym(list(ctx['vars'])[0])
        by = {}
        for a, ix, v, c in writes:
            if c: raise Opaque('conditional array write')
            by.setdefault(a, []).append((ix - iv, v))
        hi = self._exact_hi(ctx, st, L)
        for a, lst in by.items():
            offs = sorted(int(d.cval()) if d.is_const() else None for d, _ in lst)
            if offs != list(range(s)):
                raise Opaque(f'{a}: writes at offsets {offs}, expected 0..{s - 1}')
            lst = sorted(lst, key=lambda dv: dv[0].cval())
            f0 = lst[0][1]
            for d, v in lst:
                if key(shift_idx(f0, list(ctx['vars'])[0], d.cval())) != key(v):
                    raise Opaque(f'{a}: unrolled write {d} differs from write 0')
            # reads of the written array at other offsets would be a loop-carried dependence
            f = self._elementwise(f0, iv, ctx)
            v, segs = self.array_value(a, st)
            st.arr[a] = (v, segs + [Seg(ctx['lo'], hi, f)])
        for rn, _, _ in ctx['reads']:
            self.read_extent.setdefault(rn, []).append(hi)

    def _record_reads(self, ctx, st, iv, L):
        if not hasattr(self, 'read_extent'): self.read_extent = {}
        for n, ix, conds in ctx['reads']:
            hi = ix.subs({list(ctx['vars'])[0]: self._exact_hi(ctx, st, L)}) if ctx.get('step') else None
            if hi is not None and n not in self.read_extent:
                self.read_extent[n] = hi

    def _grid_map(self, writes, st, ctx, L):
        iv, jv = ctx['grid']
        nxp, nyp = ctx['gridpolys']
        kpoly = (Poly.sym(jv) - 1) * nxp + Poly.sym(iv)
        for a, ix, v, c in writes:
            if c: raise Opaque('conditional array write in 2-D nest')
            if not (ix == kpoly):
                raise Opaque(f'{a}: write index {ix} is not the column-major linearisation ({kpoly})')
            self.P.record(self.u.name, 'linearisation', f'{a}({ix}): K = ({jv}-1)*{nxp} + {iv} with '
                          f'{iv} in [1..{nxp}], {jv} in [1..{nyp}] is a bijection onto [1..{nxp * nyp}]',
                          True, line=L.line, detail='lemma (column-major linearisation)')
            def tr(x):
                if isinstance(x, ARef):
                    d = x.idx - kpoly
                    dj = d.coeff([s for s in d.syms()][0:1]) if False else None
                    # d must be di + dj*NX with constants di, dj
                    dj = 0
                    rest = d
                    if not nxp.is_const():
                        mono = list(nxp.t.keys())
                        if len(nxp.t) != 1 or nxp.t[mono[0]] != 1:
                            raise Opaque('inner extent is not a single symbol')
                        dj = d.coeff(mono[0]); rest = d - nxp * dj
                    if not rest.is_const():
                        raise Opaque(f'{x.name}({x.idx}): offset {d} is not di + dj*{nxp}')
                    di = rest.cval()
                    return ('shift', x.name, di, dj)
                return None
            def conv(e):
                def r(x):
                    t = tr(x)
                    if t is None:
                        if isinstance(x, IVal): return self.iv2ir(x.p)
                        return None
                    _, nm, di, dj = t
                    return Var(nm) if di == dj == 0 else Shift(Var(nm), di, dj)
                return rebuild(e, r)
            f = conv(v)
            # shift validity: every neighbour read must stay inside the grid under its guards
            for n, ix2, conds in ctx['reads']:
                t = tr(ARef(n, ix2))
                _, nm, di, dj = t
                if di == dj == 0: continue
                ii, jj = Poly.sym(iv) + di, Poly.sym(jv) + dj
                goal = ('and', ('and', ('rel', '>=', ii, Poly.const(1)), ('rel', '<=', ii, nxp)),
                        ('and', ('rel', '>=', jj, Poly.const(1)), ('rel', '<=', jj, nyp)))
                guards = [c[1] for c in conds if c[0] == 'icond']
                ok = self.P.valid(st.pc + ctx['range'] + guards, goal)
                self.P.record(self.u.name, 'shift-guard', f'{nm}[{iv}{di:+d},{jv}{dj:+d}] read only when inside the grid',
                              ok, line=ctx['line'], detail=' and '.join(icond_str(g) for g in guards) or 'unguarded')
                if not ok:
                    raise Opaque(f'{nm}: neighbour read may leave the grid (would wrap in linear memory)')
            vv, segs = self.array_value(a, st)
            st.arr[a] = (vv, segs + [Seg(Poly.const(1), nxp * nyp, f, grid=(str(nxp), str(nyp)))])
        for n, ix2, conds in ctx['reads']:
            self.read_extent.setdefault(n, []).append(nxp * nyp)

    # ========================================================== ORCHESTRATION MODE
    def lift_orchestration(self):
        self.mode = 'orchestration'
        stmts = self.body
        loop = self._find_while(stmts)
        self.plan_sections(stmts)
        st = State()
        out = []
        self.rt_ints = set()                 # integers that change at run time (ITER)
        if loop:
            a, b, exits, hdr_end = loop
            for v in self._assigned_ints(stmts[a:b]):
                self.rt_ints.add(v)
            out += self.emit(stmts[:a], st)
            header = stmts[a:hdr_end]
            out += self.emit(header, st)
            conds = []
            for e in exits:
                c = self.ev(e.a['cond'], st)
                conds.append(BoolOp('not', self.c2ir(c)))
            cond = conds[0]
            for c in conds[1:]: cond = BoolOp('and', cond, c)
            body = self.emit(stmts[hdr_end + len(exits):b], st) + self.emit(header, st)
            out.append(While(cond, body))
            tail = [s for s in stmts[b + 1:] if not (isinstance(s, St) and s.kind in ('continue', 'return', 'end'))]
            out += self.emit(tail, st)
        else:
            out += self.emit(stmts, st)
        self.finish_sections()
        u = self.u
        outs = {}
        for a in self.formals:
            if a not in self.writes: continue
            if a in self.sections:
                segs = self.sections[a]['order']
                outs[a] = Concat(tuple(Var(sname) for _, sname in segs))
            else:
                outs[a] = Var(a)
        used, defined = set(), set()
        def scanv(stmts):
            for x in stmts:
                if isinstance(x, Let):
                    used.update(n.name for n in nodes(x.e) if isinstance(n, Var) and n.name not in defined)
                    defined.add(x.name)
                else:
                    used.update(n.name for n in nodes(x.cond) if isinstance(n, Var) and n.name not in defined)
                    scanv(x.body)
        scanv(out)
        params = [a for a in self.formals if a not in self.sections and a in used
                  and (a in self.reads or self.is_int(a))]
        term = Def(u.name, params, out, outs)
        fp = {a: self.sections[a]['extent'] for a in self.sections}
        fp.update(self.extra_fp)
        return Summary(u.name, 'orchestration', term, params, [o for o in outs], fp,
                       [a for a in self.formals if self.is_int(a)],
                       sorted(c for c in self.commons if c in self.reads),
                       sorted(c for c in self.commons if c in self.writes))

    def finish_sections(self):
        fresh = {d[0] for d in self.P.defs}
        for arr, sec in self.sections.items():
            lens = {}
            for off, var in sec['offsets'].values():
                cands = []
                for src in sec['lengths'].get(var, []):
                    if src[0] == 'loop':
                        cands.append((src[1], 'DO loop'))
                    else:
                        _, callee, formal, hi, ints, summ = src
                        if hi is None:
                            self.unknown_fp.append(f'{callee}.{formal}')
                            self.P.record(self.u.name, 'footprint', f'{arr} section {self._segname(var)} passed to '
                                          f'{callee}.{formal}: callee footprint unknown (opaque)', False,
                                          detail='needs agent-proposed footprint (checked dynamically by canaries)')
                            continue
                        cands.append((hi.subs(ints), f'{callee}.{formal}'))
                if not cands: raise Opaque(f'{arr}: no length information for section {var}')
                def score(c):
                    sy = c.syms()
                    return (bool(sy & fresh), not sy <= set(self.formals), len(sy))
                rep = min((c for c, _ in cands), key=score)
                for c, src in cands:
                    if self.P.valid([], ('rel', '==', c, rep)): continue
                    contract = ('rel', '==', c, rep)
                    self.contracts.append({'unit': self.u.name, 'claim': f'{c} == {rep}',
                                           'why': f'section {self._segname(var)} of {arr} is used with length {c} '
                                                  f'by {src} and length {rep} elsewhere'})
                    self.P.assumptions.append(contract)
                lens[var] = rep
            items = [(off, var) for off, var in sec['offsets'].values()]
            for var, L in lens.items():
                self.P.assumptions.append(('rel', '>=', L, Poly.const(1)))
            # order sections by offset (z3), then prove they tile [1..extent] without overlap
            import functools
            def cmp(a, b):
                return -1 if self.P.valid([], ('rel', '<', a[0], b[0])) else 1
            items.sort(key=functools.cmp_to_key(cmp))
            ok = self.P.valid([], ('rel', '==', items[0][0], Poly.const(1)))
            for (o1, v1), (o2, v2) in zip(items, items[1:]):
                ok &= self.P.valid([], ('rel', '==', o2, o1 + lens[v1]))
            extent = items[-1][0] + lens[items[-1][1]] - 1
            names = [self._segname(v) for _, v in items]
            self.P.record(self.u.name, 'partition', f'{arr} = ' + ' ++ '.join(
                f'{n}[{o}..{o + lens[v] - 1}]' for (o, v), n in zip(items, names)) + f'  (disjoint, tiles [1..{extent}])',
                ok, detail='assuming section lengths >= 1')
            if not ok: raise Opaque(f'{arr}: sections do not form a partition')
            sec['order'] = [(v, n) for (o, v), n in zip(items, names)]
            sec['extent'] = extent
        cand = {}
        for nm, uses in self.name_lens.items():
            for callee, formal, hi, ints in uses:
                if hi is not None: cand.setdefault(nm, []).append(hi.subs(ints))
        for nm, his in self.read_extent.items():
            cand.setdefault(nm, []).extend(h for h in his if h is not None)
        for nm, cs in cand.items():
            cs = [c for c in cs if not (c.syms() & fresh)]
            if cs:
                self.extra_fp[nm] = min(cs, key=lambda c: (not c.syms() <= set(self.formals), len(c.syms())))

    def _find_while(self, stmts):
        labels = {s.label: i for i, s in enumerate(stmts) if isinstance(s, St) and s.label is not None}
        for b, s in enumerate(stmts):
            if isinstance(s, St) and s.kind == 'goto' and labels.get(s.a['target'], b + 1) < b:
                a = labels[s.a['target']]
                exits = [x for x in stmts[a:b] if isinstance(x, St) and x.kind == 'if_goto'
                         and labels.get(x.a['target'], -1) > b]
                if not exits: raise Opaque('backward GOTO loop without exits')
                first = stmts.index(exits[0])
                if stmts[first:first + len(exits)] != exits:
                    raise Opaque('loop exits are not consecutive')
                return a, b, exits, first
        return None

    def plan_sections(self, stmts):
        """Find hand-partitioned work arrays: formal arrays passed as A(IX)."""
        self.rt_params, self.extra_fp = set(), {}
        offs = {}
        st = State()
        def scan(sts):
            for s in sts:
                if isinstance(s, Loop): scan(s.body); continue
                if isinstance(s, If): continue
                if s.kind == 'assign' and isinstance(s.a['lhs'], Name) and self.is_int(s.a['lhs'].n):
                    v = self.ev(s.a['rhs'], st)
                    if v[0] == 'int': st.ints[s.a['lhs'].n] = v[1]
                if s.kind == 'call':
                    for x in s.a['args']:
                        if isinstance(x, Ref) and self.is_array(x.n) and len(x.args) == 1 and isinstance(x.args[0], Name):
                            off = self.ev(x.args[0], st)[1]
                            offs.setdefault(x.n, {})[str(off)] = (off, x.args[0].n)
        scan(stmts)
        self.sections = {}
        for arr, d in offs.items():
            order = sorted(d.values(), key=lambda t: str(t[0]))
            self.sections[arr] = {'offsets': d, 'lengths': {}, 'order': None}
        self._section_state = st

    def resolve_section(self, n, idx):
        if n not in self.sections: return n, idx
        for k, (off, var) in self.sections[n]['offsets'].items():
            local = idx - off + 1
            if not (local.syms() - (self.loopctx['vars'] if self.loopctx else set())):
                return self._segname(var), local
        raise Opaque(f'{n}({idx}) does not fall into a known section')

    def _segname(self, var):
        return var[1:] if var.startswith('I') and len(var) > 1 else f'W_{var}'

    def _seglen(self, arr, var, src):
        self.sections[arr]['lengths'].setdefault(var, []).append(src)

    def emit(self, stmts, st):
        out = []
        for s in stmts:
            if isinstance(s, Loop):
                self.summarize_loop(s, st)
                for a, (v, segs) in list(st.arr.items()):
                    if segs:
                        f = self.normalize_array(a, st)
                        out.append(Let(a, f)); st.arr[a] = (Var(a), [])
                        for arr, sec in self.sections.items():
                            for k, (off, var) in sec['offsets'].items():
                                if self._segname(var) == a:
                                    sec['lengths'].setdefault(var, []).append(('loop', self.footprint[a][-1]))
                continue
            if isinstance(s, If): raise Opaque('block IF in orchestration code')
            k = s.kind
            if k in ('continue', 'skip', 'return', 'end'): continue
            if k == 'assign':
                lhs = s.a['lhs']
                if not isinstance(lhs, Name): raise Opaque('array element assignment outside loop')
                n = lhs.n
                if self.is_int(n) and n not in self.rt_ints:
                    v = self.ev(s.a['rhs'], st)
                    if v[0] != 'int': raise Opaque(f'{n}: non-integer value')
                    st.ints[n] = v[1]; continue
                v = self.ev(s.a['rhs'], st)
                e = self.iv2ir(v[1]) if v[0] == 'int' else self.dval(v)
                if self.is_int(n):
                    st.ints[n] = Var(n)
                else:
                    st.dbl[n] = Var(n)
                self.note_write(n)
                out.append(Let(n, e)); continue
            if k == 'call':
                tgt, e = self.call_stmt(s, st)
                out.append(Let(tgt, e)); continue
            if k in ('if_goto', 'goto', 'logif'):
                raise Opaque(f'line {s.line}: control flow outside the recognised while-loop pattern')
            raise Opaque(f'line {s.line}: {k}')
        return out

    def _actuals(self, callee, args, st):
        cu = self.units[callee]
        summ = self.S.get(callee)
        vals, ints, targets = {}, {}, {}
        for formal, actual in zip(cu.args, args):
            fv = cu.var(formal)
            if fv.type == 'INTEGER':
                v = self.ev(actual, st)
                if v[0] != 'int': raise Opaque(f'{callee}.{formal}: run-time integer actual')
                ints[formal] = v[1]; continue
            if fv.dims:
                if isinstance(actual, Name):
                    nm = actual.n; self.note_read(nm); vals[formal] = Var(nm); targets[formal] = nm
                    hi = (summ.footprint.get(formal) if summ else None)
                    self.name_lens.setdefault(nm, []).append((callee, formal, hi, dict(ints)))
                elif isinstance(actual, Ref) and actual.n in self.sections:
                    off = self.ev(actual.args[0], st)[1]
                    var = self.sections[actual.n]['offsets'][str(off)][1]
                    sn = self._segname(var)
                    self.note_write(actual.n)
                    vals[formal] = Var(sn); targets[formal] = sn
                    hi = (summ.footprint.get(formal) if summ else None)
                    self._seglen(actual.n, var, ('call', callee, formal, hi, dict(ints), summ))
                else:
                    raise Opaque(f'{callee}.{formal}: unsupported array actual')
            else:
                v = self.ev(actual, st)
                vals[formal] = self.iv2ir(v[1]) if v[0] == 'int' else self.dval(v)
        self.call_sites.append({'callee': callee, 'ints': {k: str(v) for k, v in ints.items()},
                                'ints_poly': ints})
        return cu, summ, vals, ints, targets

    def call_expr(self, callee, args, st, is_function=True):
        if self.mode != 'orchestration':
            raise Opaque(f'call to {callee} inside a kernel')
        cu, summ, vals, ints, targets = self._actuals(callee, args, st)
        params = summ.params if summ else [f for f in cu.args if cu.var(f).type != 'INTEGER']
        return App(callee, tuple(vals[p] for p in params))

    def call_stmt(self, s, st):
        callee = s.a['name']
        cu, summ, vals, ints, targets = self._actuals(callee, s.a['args'], st)
        if summ is None or summ.kind == 'opaque':
            raise Opaque(f'CALL of opaque subroutine {callee}: outputs unknown')
        outs = [o for o in summ.outs if o != '__result__']
        if len(outs) != 1: raise Opaque(f'{callee} has {len(outs)} outputs')
        tgt = targets[outs[0]]
        if tgt in self.formals: self.note_write(tgt)
        return tgt, App(callee, tuple(vals[p] for p in summ.params))


def shift_idx(e, var, k):
    return rebuild(e, lambda x: ARef(x.name, x.idx.subs({var: Poly.sym(var) + k})) if isinstance(x, ARef) else None)


# ------------------------------------------------------------- CSE for printing
def cse(d):
    """Hoist repeated sub-expressions of the outputs into lets (keeps evaluation order)."""
    count = {}
    def walk(e):
        count[id(e)] = count.get(id(e), 0) + 1
        if count[id(e)] > 1: return
        for c in list(nodes(e))[1:2] if False else _children(e):
            walk(c)
    for e in d.outs.values(): walk(e)
    names, lets, n = {}, [], [0]
    def emit(e):
        if id(e) in names: return Var(names[id(e)])
        ch = _children(e)
        new = rebuild_children(e, [emit(c) for c in ch])
        if count.get(id(e), 0) > 1 and type(e) not in (Var, Lit, Idx):
            n[0] += 1; nm = f't{n[0]}'
            names[id(e)] = nm; lets.append(Let(nm, new)); return Var(nm)
        return new
    outs = {k: emit(v) for k, v in d.outs.items()}
    return Def(d.name, d.params, list(d.body) + lets, outs)


def _children(e):
    t = type(e)
    if t in (Bin, Cmp): return [e.a, e.b]
    if t in (Un, Shift): return [e.a]
    if t is Where: return [e.c, e.a, e.b]
    if t is Fold: return [e.body]
    if t is App: return list(e.args)
    if t is Concat: return list(e.parts)
    if t is BoolOp: return [e.a] + ([e.b] if e.b is not None else [])
    if t is ir.Elem: return [e.a, e.i]
    return []


def rebuild_children(e, ch):
    t = type(e)
    if t is Bin: return Bin(e.op, *ch)
    if t is Cmp: return Cmp(e.op, *ch)
    if t is Un: return Un(e.op, ch[0])
    if t is Shift: return Shift(ch[0], e.di, e.dj)
    if t is Where: return Where(*ch)
    if t is Fold: return Fold(ch[0])
    if t is App: return App(e.fn, tuple(ch))
    if t is Concat: return Concat(tuple(ch))
    if t is BoolOp: return BoolOp(e.op, ch[0], ch[1] if len(ch) > 1 else None)
    if t is ir.Elem: return ir.Elem(ch[0], ch[1])
    return e
