#!/usr/bin/env python3
"""MCP server exposing the lifting tools to any MCP client (Claude Code, Claude Desktop, ...).

    claude mcp add lift -- python /path/to/liftproto/mcp_server.py /path/to/workspace
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcp.server.mcpserver import MCPServer
from lift import tools

WS = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('LIFT_WS', 'ws')
GUIDE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AGENT_GUIDE.md')).read()
mcp = MCPServer('legacy-lift', instructions=GUIDE)


def _r(tool, **args):
    r = tools.call(WS, tool, {k: v for k, v in args.items() if v is not None})
    return r if isinstance(r, str) else json.dumps(r, indent=1)


@mcp.tool(description=tools.TOOLS['status'])
def status() -> str: return _r('status')

@mcp.tool(description=tools.TOOLS['worklist'])
def worklist() -> str: return _r('worklist')

@mcp.tool(description=tools.TOOLS['show'])
def show(unit: str) -> str: return _r('show', unit=unit)

@mcp.tool(description=tools.TOOLS['propose_spec'])
def propose_spec(unit: str, arg_len: dict, int_args: dict | None = None, rationale: str = '') -> str:
    return _r('propose_spec', unit=unit, arg_len=arg_len, int_args=int_args, rationale=rationale)

@mcp.tool(description=tools.TOOLS['propose_term'])
def propose_term(unit: str, term: str, rationale: str = '') -> str:
    return _r('propose_term', unit=unit, term=term, rationale=rationale)

@mcp.tool(description=tools.TOOLS['register_primitive'])
def register_primitive(unit: str, relation: str, properties: list[str] | None = None, tol: float = 1e-13,
                       rationale: str = '') -> str:
    return _r('register_primitive', unit=unit, relation=relation, properties=properties or [], tol=tol, rationale=rationale)

@mcp.tool(description=tools.TOOLS['run_probe'])
def run_probe(unit: str, inputs: dict, size: list[int] | None = None, rationale: str = '') -> str:
    return _r('run_probe', unit=unit, inputs=inputs, size=size, rationale=rationale)

@mcp.tool(description=tools.TOOLS['justify_unreachable'])
def justify_unreachable(unit: str, line: int, argument: str) -> str:
    return _r('justify_unreachable', unit=unit, line=line, argument=argument)

@mcp.tool(description=tools.TOOLS['propose_template'])
def propose_template(unit: str, name: str, template: str, bindings: dict, rationale: str = '') -> str:
    return _r('propose_template', unit=unit, name=name, template=template, bindings=bindings, rationale=rationale)

@mcp.tool(description=tools.TOOLS['annotate'])
def annotate(subject: str, name: str, meaning: str, rationale: str = '') -> str:
    return _r('annotate', subject=subject, name=name, meaning=meaning, rationale=rationale)

@mcp.tool(description=tools.TOOLS['structure'])
def structure() -> str: return _r('structure')

@mcp.tool(description=tools.TOOLS['dsl_help'])
def dsl_help() -> str: return _r('dsl_help')


if __name__ == '__main__':
    mcp.run('stdio')
