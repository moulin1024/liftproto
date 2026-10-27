"""The agent-facing tool API.  Every tool takes and returns JSON-serialisable
data, logs the call, and never executes agent-provided code."""
import json, os
from .workspace import Workspace, AgentError, structure_md, _j

DSL_HELP = open(os.path.join(os.path.dirname(__file__), 'dsl.py')).read().split('"""')[1]

TOOLS = {
    'status': 'Overview: units, verification status, fact counts, open worklist size.',
    'worklist': 'Open obligations, highest priority first. Work through these.',
    'show': 'Everything about one unit: source, lifted term, spec, verification, dynamic facts, facts.  {"unit": "DDOT"}',
    'propose_spec': 'For an opaque unit: array lengths / integer args as expressions in N, NX, NY. '
                    '{"unit", "arg_len": {"X": "N"}, "int_args": {"INCX": "1"}, "rationale"}',
    'propose_term': 'Propose a lifted term in the DSL (see dsl_help). Accepted only if bitwise-equivalent. '
                    '{"unit", "term", "rationale"}',
    'register_primitive': 'Keep an opaque unit as a native primitive with a checkable contract: a DSL relation '
                          '(R-equivalence within tol) + properties ["nonnegative", "homogeneous:k"]. '
                          '{"unit", "relation", "properties", "tol", "rationale"}',
    'run_probe': 'Run the legacy unit on explicit inputs and compare with its term / contract. Arrays: list or '
                 '{"fill": x}; omitted integers come from the spec. Units that need COMMON data (grid) take '
                 '"size": [NX, NY]. {"unit", "inputs": {"N": 3, "X": [..]}, "size": null, "rationale"}',
    'justify_unreachable': 'Argue that an uncovered branch cannot be reached (recorded as an unverified agent claim). '
                           '{"unit", "line", "argument"}',
    'propose_template': 'Claim the driver implements a known algorithm: DSL template with placeholder calls + '
                        'bindings placeholder->unit; checked by alpha-equivalence. {"unit", "name", "template", "bindings", "rationale"}',
    'annotate': 'Name / interpret a unit (recorded as an agent claim). {"subject", "name", "meaning", "rationale"}',
    'structure': 'The extracted structure document (markdown).',
    'dsl_help': 'Syntax of the term DSL.',
}


def call(ws_path, tool, args=None, actor='agent'):
    args = args or {}
    if tool not in TOOLS:
        return {'error': f'unknown tool {tool}', 'tools': TOOLS}
    ws = Workspace(ws_path)
    try:
        if tool == 'status': r = ws.status()
        elif tool == 'worklist': r = ws.state['worklist']
        elif tool == 'show': r = ws.show(args['unit'])
        elif tool == 'propose_spec': r = ws.propose_spec(args['unit'], args.get('arg_len', {}), args.get('int_args'), args.get('rationale', ''))
        elif tool == 'propose_term': r = ws.propose_term(args['unit'], args['term'], args.get('rationale', ''))
        elif tool == 'register_primitive':
            r = ws.register_primitive(args['unit'], args['relation'], args.get('properties', []),
                                      args.get('tol', 1e-13), args.get('rationale', ''))
        elif tool == 'run_probe': r = ws.run_probe(args['unit'], args['inputs'], args.get('size'), args.get('rationale', ''))
        elif tool == 'justify_unreachable': r = ws.justify_unreachable(args['unit'], args['line'], args['argument'])
        elif tool == 'propose_template':
            r = ws.propose_template(args['unit'], args['template'], args['bindings'], args['name'], args.get('rationale', ''))
        elif tool == 'annotate': r = ws.annotate(args['subject'], args['name'], args['meaning'], args.get('rationale', ''))
        elif tool == 'structure':
            r = structure_md(ws)
            open(os.path.join(ws.path, 'STRUCTURE.md'), 'w').write(r)
        elif tool == 'dsl_help': r = DSL_HELP
    except AgentError as ex:
        r = {'error': str(ex)}
    except KeyError as ex:
        r = {'error': f'missing argument {ex}'}
    summary = r if tool not in ('structure', 'dsl_help', 'show') else f'<{tool} output>'
    ws2 = Workspace(ws_path)
    ws2.log(actor, tool, args, summary)
    return json.loads(json.dumps(r, default=_j))
