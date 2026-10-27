"""Safe textual DSL for terms -- the ONLY channel through which an LLM agent can
hand a term to the checker (no Python is ever evaluated).  The syntax is exactly
what ir.pp / ir.pp_def print, so the agent reads and writes the same language.

  DDOT = λ(DX, DY).
      in  {__result__ = foldl(+, 0, DX * DY)}

Expressions:  a + b, a * b, -a, (c ? a : b), X[i-1,j] (grid shift), X(1) (element),
              F(a, b) (call of a known unit/primitive), sqrt/sin/cos/atan/abs/dble(x),
              foldl(+, 0, e), [a ++ b], comparisons, and/or/not, i, j (grid indices).
"""
import re
from . import ir
from .ir import Var, Lit, Bin, Un, Shift, Idx, Cmp, BoolOp, Where, Fold, App, Concat, Let, While, Def, Elem

TOK = re.compile(r'''\s*(?:(?P<num>\d+\.\d*(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?|\.\d+)
  |(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>\+\+|==|!=|<=|>=|[-+*/()\[\],?:<>{}=.]))''', re.X)
UNARY_FN = {'sqrt', 'sin', 'cos', 'atan', 'abs', 'dble'}
MAX_LEN = 20000


class DSLError(ValueError):
    pass


def _tok(s):
    out, i = [], 0
    while i < len(s):
        if s[i].isspace(): i += 1; continue
        m = TOK.match(s, i)
        if not m or m.end() == i: raise DSLError(f'bad character at {s[i:i+20]!r}')
        out.append((m.lastgroup, m.group(m.lastgroup))); i = m.end()
    return out


class _P:
    def __init__(self, s, fn_names):
        if len(s) > MAX_LEN: raise DSLError('expression too long')
        self.t, self.i, self.fn = _tok(s), 0, set(fn_names)

    def peek(self, k=0): return self.t[self.i + k][1] if self.i + k < len(self.t) else None
    def take(self, v=None):
        if self.i >= len(self.t): raise DSLError('unexpected end')
        x = self.t[self.i]; self.i += 1
        if v is not None and x[1] != v: raise DSLError(f'expected {v!r}, got {x[1]!r}')
        return x

    def done(self):
        if self.i != len(self.t): raise DSLError(f'trailing input at {self.t[self.i][1]!r}')

    def expr(self): return self.or_()
    def or_(self):
        a = self.and_()
        while self.peek() == 'or': self.take(); a = BoolOp('or', a, self.and_())
        return a
    def and_(self):
        a = self.not_()
        while self.peek() == 'and': self.take(); a = BoolOp('and', a, self.not_())
        return a
    def not_(self):
        if self.peek() == 'not': self.take(); return BoolOp('not', self.not_())
        return self.cmp()
    def cmp(self):
        a = self.add()
        if self.peek() in ('==', '!=', '<', '<=', '>', '>='):
            op = self.take()[1]; return Cmp(op, a, self.add())
        return a
    def add(self):
        a = self.mul()
        while self.peek() in ('+', '-'):
            op = self.take()[1]; a = Bin(op, a, self.mul())
        return a
    def mul(self):
        a = self.unary()
        while self.peek() in ('*', '/'):
            op = self.take()[1]; a = Bin(op, a, self.unary())
        return a
    def unary(self):
        if self.peek() == '-': self.take(); return Un('neg', self.unary())
        return self.postfix()
    def postfix(self):
        a = self.atom()
        while self.peek() in ('[', '(') and isinstance(a, Var):
            if self.peek() == '[':
                self.take('['); di = self._off('i'); self.take(','); dj = self._off('j'); self.take(']')
                a = Shift(a, di, dj) if (di, dj) != (0, 0) else a
            else:
                self.take('('); idx = self.expr(); self.take(')'); a = Elem(a, idx)
        return a
    def _off(self, ax):
        self.take(ax)
        if self.peek() in ('+', '-'):
            sg = 1 if self.take()[1] == '+' else -1
            k, v = self.take()
            if k != 'num' or not v.isdigit(): raise DSLError('shift offsets must be integers')
            return sg * int(v)
        return 0
    def atom(self):
        k, v = self.take()
        if k == 'num': return Lit(float(v))
        if v == '(':
            e = self.expr()
            if self.peek() == '?':
                self.take('?'); a = self.expr(); self.take(':'); b = self.expr(); self.take(')')
                return Where(e, a, b)
            self.take(')'); return e
        if v == '[':
            parts = [self.expr()]
            while self.peek() == '++': self.take(); parts.append(self.expr())
            self.take(']'); return Concat(tuple(parts))
        if k == 'name':
            if v in ('i', 'j'): return Idx(v)
            if v == 'foldl':
                self.take('('); self.take('+'); self.take(','); z = self.take()
                if z[1] not in ('0', '0.0', '0.'): raise DSLError('foldl only with (+, 0)')
                self.take(','); b = self.expr(); self.take(')'); return Fold(b)
            if v in UNARY_FN:
                self.take('('); a = self.expr(); self.take(')'); return Un(v, a)
            if v in self.fn and self.peek() == '(':
                self.take('('); args = []
                if self.peek() != ')':
                    args.append(self.expr())
                    while self.peek() == ',': self.take(); args.append(self.expr())
                self.take(')'); return App(v, tuple(args))
            return Var(v)
        raise DSLError(f'unexpected {v!r}')


def parse_expr(s, fn_names=()):
    p = _P(s, fn_names); e = p.expr(); p.done(); return e


def parse_def(text, fn_names=()):
    lines = [l.rstrip() for l in text.strip('\n').splitlines() if l.strip()]
    m = re.match(r'^\s*([A-Za-z_]\w*)\s*=\s*(?:λ|lambda)\s*\(([^)]*)\)\s*\.\s*$', lines[0])
    if not m: raise DSLError('first line must be  NAME = λ(p1, p2).')
    name, params = m.group(1), [p.strip() for p in m.group(2).split(',') if p.strip()]
    for p in params:
        if not re.fullmatch(r'[A-Za-z_]\w*', p): raise DSLError(f'bad parameter {p!r}')
    body_lines = lines[1:]
    if not body_lines or not body_lines[-1].strip().startswith('in'):
        raise DSLError('last line must be  in {OUT = expr, ...}')
    outs_txt = body_lines[-1].strip()[2:].strip()
    if not (outs_txt.startswith('{') and outs_txt.endswith('}')): raise DSLError('outputs must be {...}')
    outs = {}
    for part in _split_top(outs_txt[1:-1]):
        k, _, e = part.partition('=')
        k = k.strip()
        if not re.fullmatch(r'[A-Za-z_]\w*', k): raise DSLError(f'bad output name {k!r}')
        outs[k] = parse_expr(e, fn_names)

    def block(ls, ind):
        out, n = [], 0
        while n < len(ls):
            l = ls[n]; cur = len(l) - len(l.lstrip())
            if cur < ind: break
            s = l.strip()
            if s.startswith('let '):
                k, _, e = s[4:].partition('=')
                if not re.fullmatch(r'[A-Za-z_]\w*', k.strip()): raise DSLError(f'bad let name {k!r}')
                out.append(Let(k.strip(), parse_expr(e, fn_names))); n += 1
            elif s.startswith('while_loop'):
                c = s[len('while_loop'):].strip()
                if not c.startswith('cond:'): raise DSLError('while_loop  cond: <expr>')
                inner, used = block(ls[n + 1:], cur + 1)
                out.append(While(parse_expr(c[5:], fn_names), inner)); n += 1 + used
            else:
                raise DSLError(f'unexpected line {s!r}')
        return out, n
    body, used = block(body_lines[:-1], 1)
    if used != len(body_lines) - 1: raise DSLError('could not parse all lines (indentation?)')
    return Def(name, params, body, outs)


def _split_top(s):
    out, d, b = [], 0, ''
    for ch in s:
        d += ch in '([{'; d -= ch in ')]}'
        if ch == ',' and d == 0: out.append(b); b = ''
        else: b += ch
    if b.strip(): out.append(b)
    return out
