# Copyright (c) m8mble 2021.
# SPDX-License-Identifier: BSL-1.0

import re
import pathlib
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
from . import log


@dataclass
class Line:
    indent: str
    content: str

    def __post_init__(self):
        self.content = self.content.rstrip()

    @property
    def level(self):
        return len(self.indent)


_rx_split_line = re.compile(r"^(?P<indent>\s*)(?P<content>([^\s].*)?)$")


def split_lines(text: str) -> list[Line]:
    return [Line(**_rx_split_line.match(l).groupdict()) for l in text.splitlines()]


def load_lines(path: pathlib.Path):
    return split_lines(path.read_text())


def join_lines(lines: list[Line]) -> str:
    return "".join(f"{l.indent}{l.content}\n" for l in lines)


def write_lines(lines: list, path: pathlib.Path):
    path.write_text(join_lines(lines=lines))


@dataclass
class FilterHandler:
    forbidden: set[str] = field(default_factory=set)

    def __call__(self, lines: list[Line], **kwargs) -> list[Line]:
        return [l for l in lines if l.content not in self.forbidden]


@dataclass
class ReFilterHandler:
    forbidden: list[re.Pattern] = field(default_factory=list)

    def __call__(self, lines: list[Line], **kwargs) -> list[Line]:
        return [l for l in lines if not any(rx.match(l.content) for rx in self.forbidden)]


class SingleLineConverter:
    def __call__(self, lines: list[Line], **kwargs) -> list[Line]:
        return [self.handle_line(line, **kwargs) for line in lines]


class MultiLineConverter:
    def __call__(self, lines: list[Line], **kwargs) -> list[Line]:
        lines = [self.handle_line(l, **kwargs) for l in lines]
        return [line for l in lines for line in l] + self.handle_end(**kwargs)

    def handle_end(self, **kwargs) -> list[Line]:
        return []


class MacroCallConverter(MultiLineConverter):
    _rx_termination = re.compile(r"(?P<term>\)?;)(?P<comment>\s*(//.*|\\))?$")

    def __init__(self):
        self._macro = None
        self._buffer = []
        self._termination = {}

    def handle_line(self, line: Line, **kwargs) -> list[Line]:
        """Update state by parsing another line.

        We want to be able to find any macro invocation. This includes function-like invocations as well as more
        advanced ones which include expressions (with ';') as arguments. Therefore it's not trivial to decide when
        a macro-call is actually complete.

        The basic idea is as follows: We collect macro lines until we can assume that something else starts; it either
        starts as the current invocation really ends (among others based on indentation) or some other construct starts
        in the next line. This is why the actual conversion can be delayed by one line in certain cases.
        """
        # Check whether line continues buffered macro invocation (and in that case abort further processing)
        if self._buffer:
            if self._buffer[0].indent < line.indent or not line.content:
                self.append_to_buffer(line)
                return []
            elif self._buffer[0].indent == line.indent:
                m = self._rx_termination.match(line.content)
                if m and m.group("term").startswith(")"):
                    self.append_to_buffer(line)
                    return []

        # Handle maro invocation of previously buffered lines
        result = []
        if self._buffer:
            result = self.migrate_buffer(**kwargs)

        # Check for newly starting macro invocation
        if not self._macro:
            self._macro = self.check_start(content=line.content)
            if self._macro:
                assert line.content[len(self._macro)] == "("
                line.content = line.content[len(self._macro) + 1 :]

        if self._macro:
            self.append_to_buffer(line)
        else:
            result.append(line)

        return result

    def handle_end(self, **kwargs) -> list[Line]:
        if self._termination:
            self._buffer = self.migrate_buffer(**kwargs)
        return self._buffer

    def migrate_buffer(self, **kwargs) -> list[Line]:
        assert self._termination
        # By construction of the buffering above, there may be trailing empty lines. We artificially keep those out of
        # macro handling logic and just copy them over.
        first_empty_idx = len(self._buffer)
        while first_empty_idx > 0 and not self._buffer[first_empty_idx - 1].content:
            first_empty_idx -= 1
        empty_lines = self._buffer[first_empty_idx:]

        lines = self._buffer[:first_empty_idx]
        lines[-1].content = lines[-1].content[: -len("".join(self._termination.values()))]
        result = self.handle_macro(macro=self._macro, lines=lines, **kwargs)
        comment = self._termination.get("comment", "")
        result[-1].content += comment if comment else ""

        self._macro = None
        self._buffer = []
        self._termination = {}
        return result + empty_lines

    def append_to_buffer(self, line: Line):
        self._buffer.append(line)
        m = self._rx_termination.search(line.content)
        if m and m.group("term").startswith(")"):
            self._termination = m.groupdict(default="")


class IncludeAdder:
    def __call__(self, lines: list[Line], **kwargs) -> list[Line]:
        include_position = self._include_position(lines)
        ct_includes = [
            Line(indent="", content="#include <clean-test/clean-test.h>"),
            Line("", ""),
        ]

        ct_extras = []
        if kwargs["use_namespace_alias"]:
            ct_extras.append(Line(indent="", content="namespace ct = clean_test;"))
        if kwargs["use_literals"]:
            ct_extras.append(Line(indent="", content=f"using namespace {kwargs['namespace']}::literals;"))
        if ct_extras:
            ct_extras.append(Line("", ""))
        lines[include_position:include_position] = ct_includes + ct_extras
        return lines

    def _include_position(self, lines):
        in_multiline_comment = False
        for n, line in enumerate(lines):
            in_multiline_comment = in_multiline_comment or line.content.startswith("/*")
            if (
                line.content
                and not in_multiline_comment
                and not line.content.startswith("//")
                and not line.content.startswith("#include")
            ):
                return n
            in_multiline_comment = in_multiline_comment and not line.content.endswith("*/")
        return 0


########################################################################################################################


@dataclass
class Token:
    Kind = Enum("Kind", ["string_literal", "operator", "call_begin", "parenthesis_begin", "end", "unknown"])

    content: str
    line_idx: int
    kind: Kind


def _strip_templates(content: str) -> str:
    """(Recursively) replace template arguments by #; preserves length."""

    def replacement(m) -> str:
        return "#" * len(m.group())

    template_argument = r"\w[0-9\w_:#&]*"
    rx = rf"<\s*{template_argument}(\s,\s*{template_argument})*>"

    previous = None
    while previous != content:
        previous = content
        content = re.sub(rx, replacement, content)
    return content


def _identity(something):
    return something


def _tokenize(lines: list[Line]) -> list[Token]:
    # TODO: handling of ternary (=> some sort of if then else?)
    splitters = [
        (
            r"""(?<!\\)(u8?|U|L)?("(([^"]|(?<=\\)")*)(?<!\\)|R"(?P<eos>\w+)\(((?!\)(?P=eos)).)*\)(?P=eos))"(sv?|_\w+)?"""
            r"|"
            r"""(?<!\\)'([^']|\\.)(?<!\\)'""",
            Token.Kind.string_literal,
            _identity,
        ),
        (
            r"(\s+|\b|(^|(?<=\W))(?=\W))"
            r"(not|&&|and|\|\||or|!=?|==|<<|>>|(?<!\+)\+(?!\+)|(?<!-)-(?![->])|[<>!*/%,~.]|->)"
            r"(\s+|\b|(?<=\W)($|(?=\W)))",
            Token.Kind.operator,
            _strip_templates,
        ),
        (r"\w[\w0-9_<>\.:]*\s*[\({]", Token.Kind.call_begin, _strip_templates),
        (r"\(|{|\[", Token.Kind.parenthesis_begin, _strip_templates),
        (r"\)|}|\]", Token.Kind.end, _strip_templates),
    ]
    result = [Token(content=line.content, line_idx=l, kind=Token.Kind.unknown) for l, line in enumerate(lines)]
    for regex, kind, normalize in splitters:
        splits = []
        for t in result:
            while True:
                m = re.search(regex, normalize(t.content))
                if not m or t.kind != Token.Kind.unknown:
                    break
                start, end = m.span()
                if start > 0:
                    splits.append(Token(content=t.content[:start], kind=t.kind, line_idx=t.line_idx))
                splits.append(Token(content=m.group(), kind=kind, line_idx=t.line_idx))
                if end == len(t.content):
                    t = None
                    break
                t = Token(content=t.content[end:], kind=t.kind, line_idx=t.line_idx)
            if t:
                splits.append(t)
        result = splits
    return result


@dataclass
class Node:
    Kind = Enum("Kind", ["unary", "binary", "scope", "call", "raw"])

    parent: "Node"
    kind: Kind
    precedence: int = 0
    tokens: list[Token] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)
    lifting_barrier: bool = False
    # Whether enclosing kinds (scope / call) are already closed or currently wait for their closing parenthesis
    is_complete: bool = True


def compute_precedence(token: Token, unary: bool) -> int:
    if token.kind != Token.Kind.operator:
        return 0
    operator = token.content.strip()
    if unary and operator in {"+", "-", "~", "!", "not"}:
        return 3
    return {
        ".": 2,
        "->": 2,
        "*": 4,
        "/": 4,
        "%": 4,
        "+": 5,
        "-": 5,
        "<<": 7,
        ">>": 7,
        "<": 9,
        "<=": 9,
        ">": 9,
        ">=": 9,
        "==": 10,
        "!=": 10,
        "&&": 14,
        "and": 14,
        "||": 15,
        "or": 15,
        "=": 16,
        ",": 17,
    }[operator]


def _load_tree(tokens: list[Token]) -> Node:
    root = None
    last = None

    def make_node(**kwargs):
        nonlocal root
        node = Node(**kwargs)
        if node.parent is not None:
            node.parent.children.append(node)
        else:
            root = node
        for c in node.children:
            c.parent = node
        return node

    kind_map = {
        Token.Kind.unknown: Node.Kind.raw,
        Token.Kind.string_literal: Node.Kind.raw,
        Token.Kind.call_begin: Node.Kind.call,
        Token.Kind.parenthesis_begin: Node.Kind.scope,
    }
    for token in tokens:
        if token.kind in kind_map:
            kind = kind_map[token.kind]
            if last and kind == Node.Kind.raw and last.kind == Node.Kind.raw:  # append to last
                last.tokens.append(token)
            if (
                last
                and kind == Node.Kind.scope
                and last.kind not in {Node.Kind.unary, Node.Kind.binary}
                and last.is_complete
            ):  # chained calls
                last.tokens = list(_all_tokens(last)) + [token]
                last.kind = Node.Kind.call
                last.precedence = 0
                last.is_complete = False
            else:  # start a new last node
                last = make_node(
                    parent=last, kind=kind, precedence=0, tokens=[token], is_complete=(kind == Node.Kind.raw)
                )
        elif token.kind == Token.Kind.end:
            while last.kind not in {Node.Kind.scope, Node.Kind.call} or last.is_complete:
                last = last.parent
            last.tokens.append(token)
            last.is_complete = True
        elif token.kind == Token.Kind.operator:
            is_unary = (
                last is None
                or last.kind in {Node.Kind.unary, Node.Kind.binary}
                or (last.kind in {Node.Kind.scope, Node.Kind.call} and not last.is_complete)
            )
            precedence = compute_precedence(token, is_unary)

            parent = last
            while (
                parent is not None
                and parent.precedence < precedence
                and (parent.kind not in {Node.Kind.call, Node.Kind.scope} or parent.is_complete)
            ):
                parent = parent.parent

            if not parent:
                children = [r for r in [root] if r]
            elif is_unary:
                children = []
            else:
                children = [parent.children[-1]] if parent.children else []
                parent.children = parent.children[:-1]
            assert len(children) < 2

            last = make_node(
                parent=parent,
                kind=(Node.Kind.unary if is_unary else Node.Kind.binary),
                precedence=precedence,
                tokens=[token],
                children=children,
            )
        else:
            assert False, f"Unsupported token kind: {token}"
        display_tree(root, level="dump", title=f"Tree state after {token}")
    return root


def _all_tokens(root: Node):
    for t in root.tokens:
        yield t
    for c in root.children:
        _all_tokens(root=c)


def _lift_node(node: Node, namespace: str, **kwargs):
    if node.kind == Node.Kind.raw and len(node.tokens) == 1:
        token = node.tokens[0]
        pref = _preferential_lift_number(token.content)
        if pref:
            token.content = pref
            return

    assert node.parent is not None
    # Note: mutate node (instead of creating a new call node) to keep roots valid.
    new_node = Node(
        parent=node,
        kind=node.kind,
        precedence=node.precedence,
        tokens=node.tokens,
        children=node.children,
        lifting_barrier=node.lifting_barrier,
    )
    for c in new_node.children:
        c.parent = new_node

    call_start_line_idx, call_end_line_idx = (float("inf"), float("-inf"))
    for t in _all_tokens(root=new_node):
        call_start_line_idx = min(call_start_line_idx, t.line_idx)
        call_end_line_idx = max(call_end_line_idx, t.line_idx)

    node.kind = Node.Kind.call
    node.precedence = Node(parent=None, kind=Node.Kind.call).precedence
    node.tokens = [
        Token(content=f"{namespace}::lift(", kind=Token.Kind.call_begin, line_idx=call_start_line_idx),
        Token(content=")", kind=Token.Kind.end, line_idx=call_end_line_idx),
    ]
    node.children = [new_node]
    node.lifting_barrier = True


_preferential_lift_number_handlers = [
    # Integers
    (re.compile(r"^(?P<number>\d[\d']*)(?P<suffix>u|l|ul|ll|ull)?$"), {"": "i", "llu": "ull", "lu": "ul"}),
    # Floats
    (re.compile(r"^(?P<number>[\d']*\.[\d']+)(?P<suffix>f|ld)?$"), {"": "d"}),
]


def _preferential_lift_number(content: str) -> Optional[str]:
    """Compute preferential i.e. literal-based lift of content; return None iff this is not possible."""
    for rx, overrides in _preferential_lift_number_handlers:
        m = rx.match(content)
        if m:
            number, suffix = m.group("number"), m.groupdict(default="")["suffix"].lower()
            suffix = overrides.get(suffix, suffix)
            return f"{number}_{suffix}"


def _supports_preferential_lifting(node: Node) -> bool:
    return len(node.tokens) == 1 and _preferential_lift_number(node.tokens[0].content) is not None


def _is_member_access(node: Node) -> bool:
    return node.kind == Node.Kind.binary and any(t.content.strip() in {".", "->"} for t in node.tokens)


def _is_automatically_lifted(node: Node) -> bool:
    return node.kind in {Node.Kind.unary, Node.Kind.scope} or (
        node.kind == Node.Kind.binary and not _is_member_access(node)
    )


def _lift_tree(root: Node, **kwargs) -> Node:
    num_auto_lifted_children = 0
    children = root.children if not root.lifting_barrier else root.children[:1]
    for child in children:
        if _is_automatically_lifted(node=child) and not child.lifting_barrier:
            _lift_tree(root=child, **kwargs)
            num_auto_lifted_children += 1

    if num_auto_lifted_children == 0:
        lifting_preference = [
            # TODO: only if literals are enabled
            c
            for c in children
            if _supports_preferential_lifting(node=c) and not _is_automatically_lifted(node=c) and not c.lifting_barrier
        ]
        _lift_node(node=(lifting_preference + children)[0], **kwargs)
    return root


def _display_tree_recursive(root: Node, depth: int, level: str):
    if root is not None:
        log.log(
            f'{" " * (2 * depth + 3)} {root.precedence:02d} {root.kind} {" ".join(t.content for t in root.tokens)}'
            f"  [{hex(id(root))} -> {hex(id(root.parent)) if root.parent is not None else None}]",
            level=level,
        )
        ignored = [_display_tree_recursive(c, depth=depth + 1, level=level) for c in root.children]
    else:
        log.log("   None", level=level)


def display_tree(root: Node, level: str = "dump", title: Optional[str] = None):
    if not log.is_active(level):
        return
    if title is not None:
        log.log(title, level=level)
    _display_tree_recursive(root=root, depth=0, level=level)


def _collect_tokens(root: Node) -> list[Token]:
    if root.kind in {Node.Kind.scope, Node.Kind.call}:
        child_tokens = [_collect_tokens(root=child) for child in root.children]
        return root.tokens[:-1] + [t for ts in child_tokens for t in ts] + root.tokens[-1:]
    if root.kind == Node.Kind.raw:
        return root.tokens
    if root.kind == Node.Kind.unary:
        return [root.tokens[0]] + _collect_tokens(root=root.children[0])
    if root.kind == Node.Kind.binary:
        return _collect_tokens(root=root.children[0]) + [root.tokens[0]] + _collect_tokens(root=root.children[1])
    assert False, "Unsupported node kind!"


def _reconstruct_lines(tokens: list[Token], original: list[Line]) -> list[Line]:
    return [
        Line(content="".join(t.content for t in tokens if t.line_idx == lid), indent=line.indent)
        for lid, line in enumerate(original)
    ]


def insert_connectors(root: Node, connectors: list[str]) -> Node:
    if connectors and root.kind == Node.Kind.binary and root.tokens[-1].content.strip() == ",":
        root.tokens[-1].content = connectors[0]
        if "<<" in connectors[0]:
            root.children[1].lifting_barrier = True
        insert_connectors(root=root.children[1], connectors=connectors[1:])
    return root


def _normalize_connectors(connectors: list[str]):
    """Ensure to include the ::expect-ending braces in the first << connector."""
    matches = [i for i, c in enumerate(connectors) if "<<" in c]
    if matches:
        connectors[matches[0]] = f"){connectors[matches[0]]}"
    return bool(matches), connectors


def _connect(lines: list[Line], *, connectors: list[str] = [], **kwargs) -> list[Line]:
    assert lines
    log.log(f'Input: {"~".join(l.content for l in lines)}', level="debug")
    tokens = _tokenize(lines=lines)
    log.log(f'Tokens: {"~".join(f"{{{t.content}-{t.kind}}}" for t in tokens)}', level="debug")
    tree = _load_tree(tokens=tokens)
    display_tree(root=tree, title="Original tree without modifications", level="trace")
    tree = insert_connectors(root=tree, connectors=connectors)
    display_tree(root=tree, title="Tree with connectors", level="trace")
    return tree


def _partition_empty_prefix(lines: list[Line]) -> (list[Line], list[Line]):
    empty_lines = [l if not l.content.strip() else None for l in lines]
    empty_filtered = [l for l in empty_lines if l is not None]

    empty_prefix = [l for l, o in zip(empty_lines, empty_filtered) if l == o]
    return empty_prefix, lines[len(empty_prefix) :]


def transform_tree(lines: list[Line], *, adapter, **kwargs) -> list[Line]:
    prefix, lines = _partition_empty_prefix(lines=lines)
    assert lines
    log.debug(f'Input: {"~".join(l.content for l in lines)}')
    tokens = _tokenize(lines=lines)
    log.debug(f'Tokens: {"~".join(f"{{{t.content}-{t.kind}}}" for t in tokens)}')
    tree = _load_tree(tokens=tokens)
    display_tree(root=tree, title="Original tree without modifications", level="trace")
    tree = adapter(tree)
    display_tree(root=tree, title="Adapted tree", level="trace")
    tokens = _collect_tokens(root=tree)
    lines = prefix + _reconstruct_lines(tokens=tokens, original=lines)
    return lines


def lift(lines: list[Line], *, connectors: list[str] = [], namespace: str, **kwargs) -> list[Line]:
    expect_is_internally_closed, connectors = _normalize_connectors(connectors)

    def _adapter(tree):
        tree = insert_connectors(root=tree, connectors=connectors)

        # The brace is included for <<-connectors to close the ct::expect that will be pre-pended below.
        lift_decider = tree if tree.kind != Node.Kind.binary or ")" not in tree.tokens[-1].content else tree.children[0]
        if lift_decider.kind not in {Node.Kind.raw, Node.Kind.call} and not _is_member_access(
            lift_decider
        ):  # at least two nodes in total
            tree = _lift_tree(root=tree, namespace=namespace, **kwargs)

        display_tree(root=tree, title="Lifted Tree", level="trace")
        return tree

    try:
        lines = transform_tree(lines=lines, adapter=_adapter)
    except Exception as xcp:
        if log.is_active("trace"):
            log.exception(xcp)
        log.verbose(
            "Failed to lift the following expression, expecting a black box:\n"
            f"  {' '.join(l.content for l in lines)}",
        )

    # note: we currently assume there are no comments contained.
    lines[0].content = f"{namespace}::expect({lines[0].content}"
    if not expect_is_internally_closed:
        lines[-1].content += ")"
    lines[-1].content += ";"
    log.log(f'Output: {"~".join(l.content for l in lines)}', level="debug")
    return lines
