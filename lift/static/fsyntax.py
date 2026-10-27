"""Executable-statement parser for the F77 subset used by legacy kernels.

Produces, per program unit, a *structured* statement list:
  DO loops become Loop nodes, block IF/ELSE IF/ELSE become If nodes;
  GOTOs and labels are kept (they are resolved by the symbolic executor)."""
import re
from dataclasses import dataclass, field
from .. import f77

# ------------------------------------------------------------------ expressions
@dataclass(frozen=True)
class Num:  v: float; isint: bool
@dataclass(frozen=True)
class Name: n: str
@dataclass(frozen=True)
class Ref:  n: str; args: tuple            # array element or function / intrinsic call
@dataclass(frozen=True)
class BinX: op: str; a: object; b: object  # + - * / **
@dataclass(frozen=True)
class NegX: a: object
@dataclass(frozen=True)
class RelX: op: str; a: object; b: object  # EQ NE LT LE GT GE
@dataclass(frozen=True)
class LogX: op: str; a: object; b: object = None   # AND OR NOT

TOK = re.compile(r'''\s*(?:
   (?P<num>\d+\.(?!(?:EQ|NE|LT|LE|GT|GE|AND|OR|NOT)\.)\d*(?:[DE][+-]?\d+)?|\.\d+(?:[DE][+-]?\d+)?|\d+[DE][+-]?\d+|\d+)
 | (?P<dop>\.(?:EQ|NE|LT|LE|GT|GE|AND|OR|NOT)\.)
 | (?P<name>[A-Z][A-Z0-9_]*)
 | (?P<op>\*\*|[-+*/(),=])
)''', re.X)


def tokenize(s):
    out, i = [], 0
    s = s.strip()
    while i < len(s):
        m = TOK.match(s, i)
        if not m or m.end() == i:
            raise SyntaxError(f'cannot tokenize at: {s[i:]!r}')
        i = m.end()
        k = m.lastgroup
        out.append((k, m.group(k)))
    return out


class _P:
    def __init__(self, toks): self.t, self.i = toks, 0
    def peek(self): return self.t[self.i] if self.i < len(self.t) else (None, None)
    def next(self): tok = self.peek(); self.i += 1; return tok
    def expect(self, v):
        k, x = self.next()
        if x != v: raise SyntaxError(f'expected {v}, got {x}')

    BIN = {'.OR.': 1, '.AND.': 2, '.EQ.': 4, '.NE.': 4, '.LT.': 4, '.LE.': 4, '.GT.': 4, '.GE.': 4,
           '+': 5, '-': 5, '*': 6, '/': 6, '**': 7}

    def expr(self, minp=0):
        k, x = self.peek()
        if x == '.NOT.':
            self.next(); lhs = LogX('NOT', self.expr(3))
        elif x == '-':
            self.next(); lhs = NegX(self.expr(6))
        elif x == '+':
            self.next(); lhs = self.expr(6)
        else:
            lhs = self.atom()
        while True:
            k, x = self.peek()
            p = self.BIN.get(x)
            if p is None or p < minp or (p == minp and x != '**' and minp > 0 and False):
                break
            if p < minp:
                break
            self.next()
            rhs = self.expr(p if x == '**' else p + 1)      # ** right-assoc, others left
            if x in ('.OR.', '.AND.'):
                lhs = LogX(x[1:-1], lhs, rhs)
            elif x.startswith('.'):
                lhs = RelX(x[1:-1], lhs, rhs)
            else:
                lhs = BinX(x, lhs, rhs)
        return lhs

    def atom(self):
        k, x = self.next()
        if k == 'num':
            isint = re.fullmatch(r'\d+', x) is not None
            return Num(int(x) if isint else float(x.replace('D', 'E')), isint)
        if k == 'name':
            if self.peek()[1] == '(':
                self.next(); args = []
                if self.peek()[1] != ')':
                    while True:
                        args.append(self.expr())
                        if self.peek()[1] == ',': self.next(); continue
                        break
                self.expect(')')
                return Ref(x, tuple(args))
            return Name(x)
        if x == '(':
            e = self.expr(); self.expect(')'); return e
        raise SyntaxError(f'unexpected token {x}')


def parse_expr(s):
    p = _P(tokenize(s)); e = p.expr()
    if p.i != len(p.t): raise SyntaxError(f'trailing tokens in {s!r}')
    return e


# ------------------------------------------------------------------- statements
@dataclass
class St:
    line: int
    label: int
    kind: str                 # assign do if_goto goto logif call return continue blockif elseif else endif skip
    a: dict = field(default_factory=dict)


@dataclass
class Loop:
    line: int; label: int; var: str; lo: object; hi: object; step: object; body: list; end_label: int


@dataclass
class If:
    line: int; label: int; branches: list; els: list    # branches: [(cond, body)]


DECL = re.compile(r'^(IMPLICIT|PARAMETER|DIMENSION|COMMON|DOUBLE\s*PRECISION|INTEGER|REAL|LOGICAL|'
                  r'INTRINSIC|EXTERNAL|SAVE|DATA|FORMAT|WRITE|PRINT|READ|CHARACTER)\b')


def _paren_split(s, start):
    """s[start] == '(' ; return (inside, rest)."""
    d = 0
    for i in range(start, len(s)):
        d += s[i] == '('; d -= s[i] == ')'
        if d == 0:
            return s[start + 1:i], s[i + 1:].strip()
    raise SyntaxError(s)


def parse_stmt(line, label, s):
    if DECL.match(s) or s.startswith('SUBROUTINE') or 'FUNCTION' in s.split('(')[0] or s.startswith('PROGRAM'):
        return St(line, label, 'skip', {'text': s})
    if s in ('END',):
        return St(line, label, 'end')
    if s in ('RETURN', 'STOP'):
        return St(line, label, 'return')
    if s == 'CONTINUE':
        return St(line, label, 'continue')
    if s in ('ELSE',):
        return St(line, label, 'else')
    if s in ('END IF', 'ENDIF'):
        return St(line, label, 'endif')
    m = re.match(r'^GO\s*TO\s*(\d+)$', s)
    if m:
        return St(line, label, 'goto', {'target': int(m.group(1))})
    m = re.match(r'^DO\s+(\d+)\s+([A-Z]\w*)\s*=\s*(.*)$', s)
    if m:
        parts = _split_commas(m.group(3))
        return St(line, label, 'do', {'end': int(m.group(1)), 'var': m.group(2),
                                     'lo': parse_expr(parts[0]), 'hi': parse_expr(parts[1]),
                                     'step': parse_expr(parts[2]) if len(parts) > 2 else Num(1, True)})
    m = re.match(r'^(ELSE\s*)?IF\s*\(', s)
    if m:
        cond, rest = _paren_split(s, s.index('(', m.end() - 1))
        c = parse_expr(cond)
        if m.group(1):
            return St(line, label, 'elseif', {'cond': c})
        if rest == 'THEN':
            return St(line, label, 'blockif', {'cond': c})
        g = re.match(r'^GO\s*TO\s*(\d+)$', rest)
        if g:
            return St(line, label, 'if_goto', {'cond': c, 'target': int(g.group(1))})
        return St(line, label, 'logif', {'cond': c, 'stmt': parse_stmt(line, None, rest)})
    m = re.match(r'^CALL\s+(\w+)\s*(?:\((.*)\))?$', s)
    if m:
        args = tuple(parse_expr(a) for a in _split_commas(m.group(2) or ''))
        return St(line, label, 'call', {'name': m.group(1), 'args': args})
    m = re.match(r'^([A-Z]\w*(?:\s*\([^=]*\))?)\s*=(.*)$', s)
    if m:
        return St(line, label, 'assign', {'lhs': parse_expr(m.group(1)), 'rhs': parse_expr(m.group(2))})
    raise SyntaxError(f'line {line}: unsupported statement {s!r}')


def _split_commas(s):
    out, d, b = [], 0, ''
    for ch in s:
        d += ch == '('; d -= ch == ')'
        if ch == ',' and d == 0:
            out.append(b); b = ''
        else:
            b += ch
    if b.strip(): out.append(b)
    return out


def structure(stmts):
    """Flat statements -> nested Loop / If nodes."""
    out, i = [], 0

    def block(i, stop):
        res = []
        while i < len(stmts):
            st = stmts[i]
            if stop(st):
                return res, i
            if st.kind == 'do':
                body, j = [], i + 1
                inner, j = block(j, lambda s, e=st.a['end']: s.label == e)
                end = stmts[j]                         # terminal statement (labelled)
                if end.kind != 'continue':
                    inner.append(end)
                res.append(Loop(st.line, st.label, st.a['var'], st.a['lo'], st.a['hi'], st.a['step'],
                                inner, st.a['end']))
                i = j + 1
            elif st.kind == 'blockif':
                branches, els, cond, j = [], [], st.a['cond'], i + 1
                while True:
                    body, j = block(j, lambda s: s.kind in ('elseif', 'else', 'endif'))
                    branches.append((cond, body))
                    if stmts[j].kind == 'elseif':
                        cond = stmts[j].a['cond']; j += 1; continue
                    if stmts[j].kind == 'else':
                        els, j = block(j + 1, lambda s: s.kind == 'endif')
                    break
                res.append(If(st.line, st.label, branches, els))
                i = j + 1
            else:
                res.append(st); i += 1
        return res, i

    res, _ = block(0, lambda s: False)
    return res


def unit_bodies(path):
    """name -> structured body (excluding header/declarations/END)."""
    units = f77.parse(path)
    sts = f77.statements(path)
    bodies = {}
    for name, u in units.items():
        flat = [parse_stmt(l, b, s) for l, b, s in sts if u.line_start < l < u.line_end]
        flat = [s for s in flat if s.kind != 'skip' or s.label is not None]
        bodies[name] = structure(flat)
    return units, bodies
