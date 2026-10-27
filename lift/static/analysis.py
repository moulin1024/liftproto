"""Whole-program static analysis: lift every unit (callees first), analyse the
main program's call sites, specialise opaque units on call-site constants and
derive the dynamic-probing spec for each unit."""
from .. import f77, ir
from . import fsyntax
from .fsyntax import Ref, Name, Num, Loop, If, St, BinX
from .lifter import UnitLifter, Prover, Summary, Opaque, icond_str
from .poly import Poly


def call_graph(units, bodies):
    g = {n: set() for n in units}
    def walk_e(n, e):
        if isinstance(e, Ref):
            if e.n in units and not units[n].dims.get(e.n): g[n].add(e.n)
            for a in e.args: walk_e(n, a)
        elif hasattr(e, 'a'):
            walk_e(n, e.a)
            if getattr(e, 'b', None) is not None: walk_e(n, e.b)
    def walk(n, stmts):
        for s in stmts:
            if isinstance(s, Loop): walk(n, s.body)
            elif isinstance(s, If):
                for c, b in s.branches: walk_e(n, c); walk(n, b)
                walk(n, s.els)
            else:
                if s.kind == 'call': g[n].add(s.a['name']); [walk_e(n, a) for a in s.a['args']]
                for k in ('rhs', 'cond'):
                    if k in s.a: walk_e(n, s.a[k])
                if s.kind == 'logif' and s.a['stmt'].kind == 'assign': walk_e(n, s.a['stmt'].a['rhs'])
    for n, b in bodies.items(): walk(n, b)
    return g


def topo(g):
    out, seen = [], set()
    def visit(n):
        if n in seen: return
        seen.add(n)
        for m in sorted(g[n]): visit(m)
        out.append(n)
    for n in sorted(g): visit(n)
    return out


def main_calls(main_src):
    """Constant-propagate the main program and record its CALLs in order."""
    units, bodies = fsyntax.unit_bodies(main_src)
    prog = [u for u in units.values() if u.kind == 'PROGRAM'][0]
    env, calls = {}, []
    def val(e):
        if isinstance(e, Num): return e.v
        if isinstance(e, Name): return env.get(e.n, Name(e.n))
        if isinstance(e, BinX):
            a, b = val(e.a), val(e.b)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return {'+': a + b, '-': a - b, '*': a * b, '/': a / b}[e.op]
        return None
    for s in bodies[prog.name]:
        if isinstance(s, (Loop, If)): continue
        if s.kind == 'assign' and isinstance(s.a['lhs'], Name):
            v = val(s.a['rhs'])
            if isinstance(v, (int, float)): env[s.a['lhs'].n] = v
        if s.kind == 'call':
            calls.append({'callee': s.a['name'], 'line': s.line,
                          'args': [val(a) if not (isinstance(a, Name) and prog.dims.get(a.n)) else ('array', a.n)
                                   for a in s.a['args']]})
    return calls


def analyze(src, main_src=None):
    units, bodies = fsyntax.unit_bodies(src)
    units = {n: u for n, u in units.items() if u.kind != 'PROGRAM'}
    g = call_graph(units, bodies)
    order = topo(g)
    P = Prover()
    S, lifters, info = {}, {}, {}

    def lift_one(n, specialized=None):
        L = UnitLifter(units[n], bodies[n], units, S, P)
        if specialized: L.specialized = dict(specialized)
        try:
            s = L.lift()
        except Opaque as ex:
            s = Summary(n, 'opaque', reason=str(ex),
                        params=[a for a in units[n].args if units[n].var(a).type != 'INTEGER'],
                        outs=['__result__'] if units[n].kind == 'FUNCTION' else [],
                        int_formals=[a for a in units[n].args if units[n].var(a).type == 'INTEGER'])
        return s, L

    for n in order:
        S[n], lifters[n] = lift_one(n)

    # call sites (inside the library and from the main program)
    sites = {}
    for n, L in lifters.items():
        for cs in L.call_sites:
            sites.setdefault(cs['callee'], []).append({'caller': n, 'ints': cs['ints'], 'ints_poly': cs['ints_poly']})
    mcalls = main_calls(main_src) if main_src else []
    for c in mcalls:
        u = units.get(c['callee'])
        if not u: continue
        ints = {f: Poly.const(v) for f, v in zip(u.args, c['args'])
                if u.var(f).type == 'INTEGER' and isinstance(v, int)}
        sites.setdefault(c['callee'], []).append({'caller': 'MAIN', 'ints': {k: str(v) for k, v in ints.items()},
                                                 'ints_poly': ints, 'args': c['args']})

    # specialise opaque units on integer formals that are constant at every call site
    for n, s in list(S.items()):
        if s.kind != 'opaque': continue
        consts = {}
        for f in units[n].args:
            if units[n].var(f).type != 'INTEGER': continue
            vals = {str(cs['ints_poly'].get(f)) for cs in sites.get(n, []) if f in cs['ints_poly']}
            if len(vals) == 1 and sites.get(n):
                v = next(iter(vals))
                if v.lstrip('-').isdigit(): consts[f] = int(v)
        if consts:
            s2, L2 = lift_one(n, consts)
            s2.specialized = consts
            if s2.kind == 'opaque':
                s.reason_specialized = s2.reason
            s.specialized = consts
            if s2.kind != 'opaque':
                S[n], lifters[n] = s2, L2

    # branch / loop facts
    branches, loops = [], []
    for n, L in lifters.items():
        lines = {}
        for (line, side), feas in L.branch_seen.items():
            lines.setdefault(line, {})[side] = feas
        for line, d in sorted(lines.items()):
            for side in (True, False):
                if side in d and not d[side]:
                    branches.append({'unit': n, 'line': line, 'branch': 'taken' if side else 'not taken',
                                     'status': 'infeasible (proved)'})
                if side not in d:
                    branches.append({'unit': n, 'line': line, 'branch': 'taken' if side else 'not taken',
                                     'status': 'infeasible (proved: pruned on every path)'})
        for line, zt in sorted(L.loop_ztrip.items()):
            loops.append({'unit': n, 'line': line, 'zero_trip': 'feasible' if zt else 'infeasible (proved)'})
            if not zt:
                P.record(n, 'zero-trip', f'DO loop at line {line} executes at least once on every path', True, line=line)

    contracts = [c for L in lifters.values() for c in L.contracts]
    for n, cs in sites.items():
        u = units[n]
        for f in u.args:
            if u.var(f).type != 'INTEGER': continue
            vals = {cs_['ints'].get(f) for cs_ in cs if f in cs_['ints']}
            lib = [c for c in cs if c['caller'] != 'MAIN']
            vals = {c['ints'].get(f) for c in lib if f in c['ints']}
            if lib and len(vals) == 1 and None not in vals and next(iter(vals)).lstrip('-').isdigit():
                contracts.append({'unit': n, 'claim': f'{f} == {next(iter(vals))}',
                                  'why': f'constant at all {len(lib)} call site(s) inside the library'})
    # N == NX*NY check at the main call site of the driver
    for c in mcalls:
        if c['callee'] in S and S[c['callee']].kind == 'orchestration':
            for k in contracts:
                if k['unit'] == c['callee']:
                    k['main_call_site'] = check_contract_main(k, c, units, mcalls)

    specs = {n: derive_spec(n, S, units, sites, mcalls) for n in S}
    return {'units': units, 'summaries': S, 'lifters': lifters, 'prover': P, 'sites': sites,
            'main_calls': mcalls, 'branches': branches, 'loops': loops, 'contracts': contracts,
            'specs': specs, 'order': order, 'callgraph': {k: sorted(v) for k, v in g.items()}}


def check_contract_main(k, call, units, mcalls):
    u = units[call['callee']]
    env = {f: v for f, v in zip(u.args, call['args']) if isinstance(v, int)}
    for c in mcalls:                                  # COMMON NX/NY set by a SETUP-like call
        if c['callee'] == 'SETUP':
            su = units['SETUP']
            vals = dict(zip(su.args, c['args']))
            env.setdefault('NX', vals.get('NXI')); env.setdefault('NY', vals.get('NYI'))
    try:
        lhs, rhs = k['claim'].split('==')
        ok = eval(lhs, {}, env) == eval(rhs, {}, env)
        return f'holds at MAIN line {call["line"]} ({env})' if ok else f'VIOLATED at MAIN line {call["line"]}'
    except Exception as ex:
        return f'unknown ({ex})'


def derive_spec(n, S, units, sites, mcalls):
    s, u = S[n], units[n]
    spec = {'arg_len': {}, 'int_args': {}, 'common_int': {}, 'common_len': {}, 'source': {}}
    fp = dict(s.footprint)
    consts = getattr(s, 'specialized', {}) or {}
    lib_sites = [c for c in sites.get(n, []) if c['caller'] != 'MAIN']
    main_sites = [c for c in sites.get(n, []) if c['caller'] == 'MAIN']
    grid_formals = {}
    if s.term is not None:                                   # COMMON NX := formal NXI  ->  NXI is the grid size
        for o, e in s.term.outs.items():
            if isinstance(e, ir.Var) and e.name in u.args and o in ('NX', 'NY'):
                grid_formals[e.name] = o
    size_syms = set()
    for h in fp.values(): size_syms |= h.syms()
    for a in u.args:
        v = u.var(a)
        if v.type == 'INTEGER':
            pv = {str(c['ints_poly'].get(a)) for c in lib_sites if a in c['ints_poly']}
            if a in grid_formals:
                val, src = grid_formals[a], 'grid extent (writes COMMON %s)' % grid_formals[a]
            elif a in consts:
                val, src = str(consts[a]), 'constant at every call site'
            elif len(pv) == 1 and next(iter(pv)).lstrip('-').isdigit():
                val, src = next(iter(pv)), 'constant at every call site'
            elif a in size_syms or pv == {'N'}:
                val, src = 'N', 'size (appears in footprints / passed the caller size)'
            elif s.kind == 'orchestration' and main_sites and a != 'N':
                mv = [x for f, x in zip(u.args, main_sites[0]['args']) if f == a]
                val, src = (str(mv[0]), 'MAIN call-site constant') if mv and isinstance(mv[0], int) else ('0', 'output only')
            elif a == 'N':
                val, src = 'N', 'size (by name -- heuristic)'
            else:
                val, src = '0', 'unknown (default 0)'
            spec['int_args'][a] = val; spec['source'][a] = src
        elif v.dims:
            if a in fp:
                h = fp[a].subs({k: Poly.const(c) for k, c in consts.items()})
                spec['arg_len'][a] = str(h); spec['source'][a] = 'static footprint'
            else:
                spec['arg_len'][a] = None; spec['source'][a] = 'UNKNOWN (opaque) -- agent must propose'
    for blk, names in u.commons.items():
        for c in names:
            cv = u.var(c)
            if cv.type == 'INTEGER':
                spec['common_int'][f'{blk}.{c}'] = c
            elif cv.dims and c in fp and c not in s.writes_common:
                spec['common_len'][f'{blk}.{c}'] = str(fp[c])
    if s.writes_common:
        spec['common_out'] = [f'{blk}.{c}' for blk, names in u.commons.items() for c in names if c in s.writes_common]
    if s.kind == 'orchestration':
        for c in mcalls:
            if c['callee'] != n: continue
            spec['fixed'] = {f: v for f, v in zip(u.args, c['args'])
                             if u.var(f).type != 'INTEGER' and not u.var(f).dims and isinstance(v, float)}
            prev = [p for p in mcalls if p['line'] < c['line'] and p['callee'] in S and S[p['callee']].writes_common]
            if prev:
                su = S[prev[-1]['callee']]
                spec['setup'] = (su.name, dict(derive_spec(su.name, S, units, sites, mcalls)['int_args']))
            spec['common_len'] = {}
    return spec
