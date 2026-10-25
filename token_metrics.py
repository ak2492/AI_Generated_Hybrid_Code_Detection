"""Tree-sitter source-token accounting shared by every timed evaluation path.

The metric is intentionally model-agnostic: one token is one leaf in the
language's tree-sitter parse (keywords, identifiers, literals, operators,
punctuation and comments all count; whitespace does not).  This keeps
tokens/sec comparable across the hybrid, CPG and legacy detectors even though
they consume different feature representations.
"""

from __future__ import annotations

_PARSERS = {}


def _get_parser(language):
    parser = _PARSERS.get(language)
    if parser is not None:
        return parser

    from tree_sitter import Language, Parser

    if language == "python":
        import tree_sitter_python as tsmod
    elif language == "java":
        import tree_sitter_java as tsmod
    elif language == "cpp":
        import tree_sitter_cpp as tsmod
    else:
        raise ValueError(f"Unsupported language: {language}")

    lang = Language(tsmod.language())
    try:
        parser = Parser(lang)
    except TypeError:
        parser = Parser()
        parser.set_language(lang)
    _PARSERS[language] = parser
    return parser


def count_tree_tokens(root):
    """Count leaf nodes under a tree-sitter root node."""
    if root is None:
        return 0
    total = 0
    stack = [root]
    while stack:
        node = stack.pop()
        children = node.children
        if children:
            stack.extend(children)
        else:
            total += 1
    return total


def count_code_tokens(code, language):
    """Count source tokens for one snippet."""
    if not code or not isinstance(code, str):
        return 0
    parser = _get_parser(language)
    tree = parser.parse(code.encode("utf8", errors="ignore"))
    return count_tree_tokens(tree.root_node)


def count_tokens_batch(codes, language):
    """Count source tokens for a sequence of snippets."""
    parser = _get_parser(language)
    counts = []
    for code in codes:
        if not code or not isinstance(code, str):
            counts.append(0)
            continue
        tree = parser.parse(code.encode("utf8", errors="ignore"))
        counts.append(count_tree_tokens(tree.root_node))
    return counts
