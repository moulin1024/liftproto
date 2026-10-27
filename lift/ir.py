"""A tiny functional combinator IR with an *order-preserving* interpreter.

Every node has a fixed IEEE evaluation order, so a lifted term can be replayed
bitwise against the legacy code.  Reductions are explicit sequential left folds
(numpy's np.sum is pairwise and would NOT be bitwise-equal)."""
from dataclasses import dataclass
import numpy as np


class E:
    def __add__(s, o): return Bin('+', s, _l(o))
    def __radd__(s, o): return Bin('+', _l(o), s)
    def __sub__(s, o): return Bin('-', s, _l(o))
    def __rsub__(s, o): return Bin('-', _l(o), s)
    def __mul__(s, o): return Bin('*', s, _l(o))
    def __rmul__(s, o): return Bin('*', _l(o), s)
    def __truediv__(s, o): return Bin('/', s, _l(o))
    def __rtruediv__(s, o): return Bin('/', _l(o), s)
    def __neg__(s): return Un('neg', s)
    def __str__(s): return pp(s)


def _l(o): return o if isinstance(o, E) else Lit(float(o))


@dataclass(frozen=True, eq=False)
class Var(E): name: str
@dataclass(frozen=True, eq=False)
class Lit(E): v: float
@dataclass(frozen=True, eq=False)
class Bin(E): op: str; a: E; b: E
@dataclass(frozen=True, eq=False)
class Un(E): op: str; a: E                         # 'neg' | 'sqrt'
@dataclass(frozen=True, eq=False)
class Shift(E): a: E; di: int; dj: int             # a[i+di, j+dj], zero outside grid
@dataclass(frozen=True, eq=False)
class Idx(E): axis: str                            # 1-based grid index 'i' | 'j'
@dataclass(frozen=True, eq=False)
class Cmp(E): op: str; a: E; b: E
@dataclass(frozen=True, eq=False)
class BoolOp(E): op: str; a: E; b: E = None        # 'and' | 'or' | 'not'
@dataclass(frozen=True, eq=False)
class Where(E): c: E; a: E; b: E
@dataclass(frozen=True, eq=False)
class Fold(E): body: E                             # foldl (+) 0.0, sequential
@dataclass(frozen=True, eq=False)
class App(E): fn: str; args: tuple
@dataclass(frozen=True, eq=False)
class Concat(E): parts: tuple
@dataclass(frozen=True, eq=False)
class Elem(E): a: E; i: E                          # a(i), 1-based scalar element


@dataclass
class Let: name: str; e: E
@dataclass
class While: cond: E; body: list


@dataclass
class Def:
    name: str
    params: list
    body: list          # [Let | While]
    outs: dict          # output name -> E


I, J = Idx('i'), Idx('j')
UN = {'neg': np.negative, 'sqrt': np.sqrt, 'sin': np.sin, 'cos': np.cos, 'atan': np.arctan,
      'abs': np.abs, 'dble': lambda a: np.asarray(a, dtype=np.float64)}
HOOK = {'app': None}           # optional tracer: called as hook(fn, arg_values, env)
NX, NY = Var('NX'), Var('NY')


# ---------------------------------------------------------------- interpreter
def ev(e, env, fns):
    t = type(e)
    if t is Var: return env[e.name]
    if t is Lit: return e.v
    if t is Bin:
        a, b = ev(e.a, env, fns), ev(e.b, env, fns)
        return {'+': np.add, '-': np.subtract, '*': np.multiply, '/': np.divide}[e.op](a, b)
    if t is Un:
        a = ev(e.a, env, fns)
        return UN[e.op](a)
    if t is Shift:
        a = np.asarray(ev(e.a, env, fns)); nx, ny = env['NX'], env['NY']
        g = a[:nx * ny].reshape(ny, nx); out = np.zeros_like(g)
        js, je = max(0, -e.dj), min(ny, ny - e.dj); is_, ie = max(0, -e.di), min(nx, nx - e.di)
        out[js:je, is_:ie] = g[js + e.dj:je + e.dj, is_ + e.di:ie + e.di]
        return out.reshape(-1)
    if t is Idx:
        nx, ny = env['NX'], env['NY']
        k = np.arange(nx * ny)
        return (k % nx + 1) if e.axis == 'i' else (k // nx + 1)
    if t is Cmp:
        a, b = ev(e.a, env, fns), ev(e.b, env, fns)
        return {'>': np.greater, '<': np.less, '<=': np.less_equal, '>=': np.greater_equal,
                '==': np.equal, '!=': np.not_equal}[e.op](a, b)
    if t is BoolOp:
        a = ev(e.a, env, fns)
        if e.op == 'not': return np.logical_not(a)
        return (np.logical_and if e.op == 'and' else np.logical_or)(a, ev(e.b, env, fns))
    if t is Where:
        c = ev(e.c, env, fns)
        if np.ndim(c) == 0:               # scalar branch: evaluate lazily
            return ev(e.a, env, fns) if c else ev(e.b, env, fns)
        return np.where(c, ev(e.a, env, fns), ev(e.b, env, fns))
    if t is Fold:
        b = np.asarray(ev(e.body, env, fns), dtype=np.float64)
        return float(np.add.accumulate(np.concatenate([[0.0], b]))[-1])
    if t is App:
        d = fns[e.fn]
        vals = [ev(a, env, fns) for a in e.args]
        if HOOK['app'] is not None:
            HOOK['app'](e.fn, vals, env)
        if getattr(d, 'native', None) is not None:        # opaque primitive: run the legacy code
            return d.native(vals, env)
        r = run(d, dict(zip(d.params, vals)), fns, env)
        return next(iter(r.values())) if len(r) == 1 else r
    if t is Concat:
        return np.concatenate([np.atleast_1d(ev(p, env, fns)) for p in e.parts])
    if t is Elem:
        return np.asarray(ev(e.a, env, fns))[int(ev(e.i, env, fns)) - 1]
    raise TypeError(t)


def _exec(stmts, env, fns):
    for s in stmts:
        if isinstance(s, Let):
            env[s.name] = ev(s.e, env, fns)
        else:
            while ev(s.cond, env, fns):
                _exec(s.body, env, fns)


def run(d, args, fns, outer=None):
    env = {k: outer[k] for k in ('NX', 'NY', 'N') if outer and k in outer}
    env.update(outer.get('__globals__', {}) if outer else {})
    env.update(args)
    if outer is not None and '__globals__' in outer:
        env['__globals__'] = outer['__globals__']
    _exec(d.body, env, fns)
    return {k: ev(e, env, fns) for k, e in d.outs.items()}


# ------------------------------------------------------------- pretty printer
PREC = {'or': 1, 'and': 2, 'cmp': 3, '+': 4, '-': 4, '*': 5, '/': 5}


def pp(e, p=0):
    t = type(e)
    if t is Var: return e.name
    if t is Lit:
        return repr(e.v).rstrip('0').rstrip('.') if e.v == int(e.v) else repr(e.v)
    if t is Bin:
        q = PREC[e.op]
        s = f'{pp(e.a, q)} {e.op} {pp(e.b, q + 1)}'      # left-assoc: right operand binds tighter
        return f'({s})' if q < p else s
    if t is Un:
        return f'-{pp(e.a, 9)}' if e.op == 'neg' else f'{e.op}({pp(e.a)})'
    if t is Shift:
        f = lambda ax, d: ax if d == 0 else f'{ax}{d:+d}'
        return f'{pp(e.a, 9)}[{f("i", e.di)},{f("j", e.dj)}]'
    if t is Idx: return e.axis
    if t is Cmp:
        s = f'{pp(e.a, 4)} {e.op} {pp(e.b, 4)}'
        return f'({s})' if PREC['cmp'] < p else s
    if t is BoolOp:
        if e.op == 'not': return f'not {pp(e.a, 9)}'
        q = PREC[e.op]; s = f'{pp(e.a, q)} {e.op} {pp(e.b, q)}'
        return f'({s})' if q < p else s
    if t is Where: return f'({pp(e.c)} ? {pp(e.a)} : {pp(e.b)})'
    if t is Fold: return f'foldl(+, 0, {pp(e.body)})'
    if t is App: return f'{e.fn}({", ".join(pp(a) for a in e.args)})'
    if t is Concat: return '[' + ' ++ '.join(pp(x) for x in e.parts) + ']'
    if t is Elem: return f'{pp(e.a, 9)}({pp(e.i)})'
    raise TypeError(t)


def pp_def(d):
    lines = [f'{d.name} = λ({", ".join(d.params)}).']
    def emit(stmts, ind):
        for s in stmts:
            if isinstance(s, Let):
                lines.append(f'{ind}let {s.name} = {pp(s.e)}')
            else:
                lines.append(f'{ind}while_loop  cond: {pp(s.cond)}')
                emit(s.body, ind + '    ')
    emit(d.body, '    ')
    outs = ', '.join(f'{k} = {pp(e)}' for k, e in d.outs.items())
    lines.append(f'    in  {{{outs}}}')
    return '\n'.join(lines)
