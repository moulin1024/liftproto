"""Minimal fixed-form Fortran 77 *interface* parser.

Only extracts what the dynamic harness needs to call a routine: unit names,
argument lists, types (incl. IMPLICIT rules), dimensions, PARAMETERs and
COMMON block layouts.  It deliberately does NOT analyse executable statements:
all semantics are recovered dynamically.
"""
import re
from dataclasses import dataclass, field


@dataclass
class Var:
    name: str
    type: str                 # 'INTEGER' | 'DOUBLE PRECISION' | 'REAL' | 'LOGICAL'
    dims: list = field(default_factory=list)   # list of dimension strings


@dataclass
class Unit:
    name: str
    kind: str                 # 'SUBROUTINE' | 'FUNCTION' | 'PROGRAM'
    args: list
    rtype: str = None
    line_start: int = 0
    line_end: int = 0
    params: dict = field(default_factory=dict)
    commons: dict = field(default_factory=dict)   # block -> [names]
    explicit: dict = field(default_factory=dict)  # name -> type
    dims: dict = field(default_factory=dict)      # name -> [dims]
    implicit: dict = field(default_factory=dict)  # letter -> type

    def var(self, name):
        t = self.explicit.get(name)
        if t is None:
            t = self.implicit.get(name[0], 'INTEGER' if name[0] in 'IJKLMN' else 'REAL')
        return Var(name, t, self.dims.get(name, []))

    def eval_dim(self, d):
        return int(eval(d, {}, dict(self.params)))


def _statements(path):
    return [(l, s) for l, _, s in statements(path)]


def statements(path):
    """(line, label, UPPERCASE text) for every statement, continuation lines joined."""
    stmts, cur, cur_line, cur_lab = [], None, 0, None
    with open(path) as f:
        for ln, raw in enumerate(f, 1):
            raw = raw.rstrip('\n')
            if not raw.strip() or raw[0] in 'Cc*!':
                continue
            raw = raw[:72].ljust(7)
            if raw[5] not in ' 0' and cur is not None:   # continuation line
                cur += ' ' + raw[6:].strip()
                continue
            if cur is not None:
                stmts.append((cur_line, cur_lab, cur))
            cur, cur_line = raw[6:].strip(), ln
            cur_lab = int(raw[:5]) if raw[:5].strip() else None
    if cur is not None:
        stmts.append((cur_line, cur_lab, cur))
    return [(l, b, s.upper()) for l, b, s in stmts]


def _split_top(s):
    out, depth, buf = [], 0, ''
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(buf.strip()); buf = ''
        else:
            buf += ch
    if buf.strip():
        out.append(buf.strip())
    return out


def _entity(e):
    m = re.match(r'(\w+)\s*(?:\((.*)\))?$', e.strip())
    return m.group(1), ([d.strip() for d in _split_top(m.group(2))] if m.group(2) else [])


TYPES = r'(DOUBLE\s*PRECISION|INTEGER|REAL|LOGICAL)'


def parse(path):
    units, u, last_line = {}, None, 0
    for line, s in _statements(path):
        last_line = line
        m = re.match(rf'^(?:{TYPES}\s+)?FUNCTION\s+(\w+)\s*\((.*)\)$', s)
        if m:
            u = Unit(m.group(2), 'FUNCTION', [a.strip() for a in m.group(3).split(',')],
                     rtype=re.sub(r'\s+', ' ', m.group(1)) if m.group(1) else None, line_start=line)
            continue
        m = re.match(r'^SUBROUTINE\s+(\w+)\s*\((.*)\)$', s)
        if m:
            u = Unit(m.group(1), 'SUBROUTINE', [a.strip() for a in m.group(2).split(',')], line_start=line)
            continue
        m = re.match(r'^PROGRAM\s+(\w+)$', s)
        if m:
            u = Unit(m.group(1), 'PROGRAM', [], line_start=line)
            continue
        if u is None:
            continue
        if s == 'END':
            u.line_end = line
            if u.kind == 'FUNCTION' and u.rtype:
                u.explicit.setdefault(u.name, u.rtype)
            units[u.name] = u
            u = None
            continue
        m = re.match(rf'^IMPLICIT\s+{TYPES}\s*\((.*)\)$', s)
        if m:
            t = re.sub(r'\s+', ' ', m.group(1))
            for rng in m.group(2).split(','):
                a, _, b = rng.strip().partition('-')
                for c in range(ord(a), ord(b or a) + 1):
                    u.implicit[chr(c)] = t
            continue
        m = re.match(r'^PARAMETER\s*\((.*)\)$', s)
        if m:
            for kv in _split_top(m.group(1)):
                k, v = kv.split('=')
                u.params[k.strip()] = eval(v.replace('D', 'E'), {}, dict(u.params))
            continue
        m = re.match(r'^DIMENSION\s+(.*)$', s)
        if m:
            for e in _split_top(m.group(1)):
                n, d = _entity(e); u.dims[n] = d
            continue
        m = re.match(r'^COMMON\s*/\s*(\w+)\s*/\s*(.*)$', s)
        if m:
            names = []
            for e in _split_top(m.group(2)):
                n, d = _entity(e)
                if d:
                    u.dims[n] = d
                names.append(n)
            u.commons[m.group(1)] = names
            continue
        m = re.match(rf'^{TYPES}\s+(.*)$', s)
        if m and 'FUNCTION' not in s:
            t = re.sub(r'\s+', ' ', m.group(1))
            for e in _split_top(m.group(2)):
                n, d = _entity(e)
                u.explicit[n] = t
                if d:
                    u.dims[n] = d
    return units
