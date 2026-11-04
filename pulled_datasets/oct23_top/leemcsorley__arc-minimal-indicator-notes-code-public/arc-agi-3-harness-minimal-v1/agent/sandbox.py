"""The model's virtual code file (py / patch tools), its safety scanner and output helpers.

Analysis code (the `run` tool) executes in act_sandbox.run_interactive, not here."""
from __future__ import annotations

import ast
import builtins as _b
from dataclasses import dataclass
import re
from typing import Any, Callable

import numpy as np


FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "__import__", "globals",
    "vars", "locals", "getattr", "setattr", "delattr", "input",
    "breakpoint", "memoryview", "help", "__build_class__",
}


ALLOWED_IMPORT_MODULES = {"numpy", "np", "collections", "itertools", "functools", "math", "heapq", "bisect", "re", "copy",
                          "json", "operator", "statistics", "fractions", "string", "random", "time"}
# v2.8: every allowed module is already imported under its own name (numpy as np) in run and in the code file
PREIMPORTED_MODULES = ("collections", "itertools", "functools", "math", "heapq", "bisect", "re", "copy", "json",
                       "operator", "statistics", "fractions", "string", "random", "time")
ALLOWED_MODULES_TEXT = ("numpy (as np), " + ", ".join(PREIMPORTED_MODULES))


def preimported_namespace() -> dict[str, Any]:
    """The allowed modules, imported once, for the code file's namespace."""
    import importlib
    out: dict[str, Any] = {"np": np}
    for _m in PREIMPORTED_MODULES:
        try:
            out[_m] = importlib.import_module(_m)
        except Exception:
            pass
    return out


class Scanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.hits: list[tuple[str, int | str]] = []

    def flag(self, node: ast.AST, why: str) -> None:
        self.hits.append((why, getattr(node, "lineno", "?")))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            mod_root = alias.name.split(".")[0]
            if mod_root not in ALLOWED_IMPORT_MODULES:
                self.flag(node, f"import of '{mod_root}' is not allowed")
                break
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod_root = (node.module or "").split(".")[0]
        if mod_root not in ALLOWED_IMPORT_MODULES:
            self.flag(node, f"import of '{mod_root}' is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.flag(node, f"dunder attribute .{node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            self.flag(node, f"dunder name {node.id}")
        elif node.id in FORBIDDEN_CALLS:
            self.flag(node, f"forbidden builtin {node.id}")
        self.generic_visit(node)


def safety_error(hits: list[tuple[str, int | str]], max_hits: int = 5, prefix: str = "Safety check failed") -> str:
    """v2.8: the error text for a refused snippet / code file. A refused import names the module and lists the
    allowed ones (all already imported), and says that nothing ran."""
    details = "; ".join(f"line {ln}: {why}" for why, ln in hits[:max_hits])
    msg = f"{prefix}: {details}"
    mods = sorted({w.split("'")[1] for w, _ in hits if w.startswith("import of '")})
    if mods:
        own = [m for m in mods if m in RUN_NAMESPACE_NAMES]
        msg += (". Nothing in this code ran. Allowed modules, all already imported under their own names (no import "
                f"needed): {ALLOWED_MODULES_TEXT}. The recursion limit is already raised (no sys needed).")
        if own:
            msg += " " + ", ".join(own) + (" is" if len(own) == 1 else " are") + " already available in run as a function (no import needed)."
    return msg


def is_safe(code: str) -> tuple[bool, list[tuple[str, int | str]]]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, [(f"syntax error: {exc}", "?")]
    scanner = Scanner()
    scanner.visit(tree)
    return (len(scanner.hits) == 0), scanner.hits


_SAFE_BUILTIN_NAMES = [
    "abs", "min", "max", "sum", "round", "len", "range", "enumerate",
    "zip", "map", "filter", "sorted", "reversed", "all", "any",
    "int", "float", "bool", "str", "list", "tuple", "dict", "set",
    "frozenset", "print", "isinstance", "divmod", "pow", "slice",
    "next", "iter", "repr", "type", "bin", "hex", "oct", "ord", "chr", "format", "callable", "hash",
    "hasattr", "id", "issubclass", "object",
    "True", "False", "None", "Exception", "ValueError", "TypeError",
    "KeyError", "IndexError", "RuntimeError", "StopIteration",
    "ZeroDivisionError", "AssertionError", "AttributeError", "NameError", "NotImplementedError",
    "ArithmeticError", "LookupError", "OverflowError", "RecursionError",
]
SAFE_BUILTINS = {name: getattr(_b, name) for name in _SAFE_BUILTIN_NAMES if hasattr(_b, name)}


def _guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
    if name.split(".")[0] not in ALLOWED_IMPORT_MODULES:
        raise ImportError(f"import of '{name}' is not allowed")
    return _b.__import__(name, *args, **kwargs)


SAFE_BUILTINS["__import__"] = _guarded_import


def _try_parse_numpy_row(row_text: str) -> list[str] | None:
    """Try to parse a single row of numpy array text into numeric strings."""
    stripped = row_text.strip().strip("[]")
    if not stripped:
        return None
    tokens = stripped.split()
    for tok in tokens:
        tok_clean = tok.rstrip(",")
        if not tok_clean.lstrip("-").replace(".", "", 1).isdigit():
            return None
    return [tok.rstrip(",") for tok in tokens]


def _postprocess_numpy_to_csv(output: str) -> tuple[str, bool]:
    """Convert numpy 2D array repr in stdout to spaceless CSV.
    Returns (processed_output, had_conversions)."""
    lines = output.split("\n")
    result_lines: list[str] = []
    converted = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("[["):
            block_lines = [line]
            while not block_lines[-1].rstrip().endswith("]]"):
                i += 1
                if i >= len(lines):
                    break
                block_lines.append(lines[i])

            block_text = "\n".join(block_lines)
            inner = block_text.strip()
            if inner.startswith("[[") and inner.endswith("]]"):
                inner_content = inner[1:-1]  # strip outer [ ]
                row_texts = re.split(r"\]\s*\n\s*\[", inner_content)
                csv_rows: list[str] = []
                parse_ok = True
                for rt in row_texts:
                    parsed = _try_parse_numpy_row(rt)
                    if parsed is None:
                        parse_ok = False
                        break
                    csv_rows.append(",".join(parsed))
                if parse_ok and csv_rows:
                    result_lines.extend(csv_rows)
                    converted = True
                else:
                    result_lines.extend(block_lines)
            else:
                result_lines.extend(block_lines)
        else:
            result_lines.append(line)
        i += 1

    return "\n".join(result_lines), converted


@dataclass
class VirtualCodeFile:
    """Manages the model's virtual Python file. v2.11: with a loader set (harness_ext.code_loader) the file is loaded in a
    sandbox worker process to check it, never in the harness process; namespace stays None."""
    source: str = ""
    namespace: dict[str, Any] | None = None
    compile_error: str | None = None
    loader: Callable[[str], str | None] | None = None

    def update_full(self, code: str) -> str | None:
        """Replace entire source. Returns error message or None on success."""
        safe, hits = is_safe(code)
        if not safe:
            return safety_error(hits, 5)
        return self._try_source(code)

    def _try_source(self, new_source: str) -> str | None:
        """v2.10: load the new text into a fresh namespace first; the file is replaced only when that succeeds.
        v2.11: loaded by the loader (in a worker process) when one is set."""
        if self.loader is not None:
            err = self.loader(new_source)
            if err:
                return (f"Not applied: loading the new code failed: {err} (top-level statements run when the file is loaded, "
                        "before states/current_board exist). Your previous code file is unchanged and still active.")
            self.source = new_source
            self.namespace = None
            self.compile_error = None
            return None
        namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS, **preimported_namespace()}
        try:
            exec(new_source, namespace)
        except Exception as exc:
            return (f"Not applied: loading the new code raised {exc!r} (top-level statements run when the file is loaded, "
                    "before states/current_board exist). Your previous code file is unchanged and still active.")
        self.source = new_source
        self.namespace = namespace
        self.compile_error = None
        return None

    def has(self, name: str) -> bool:
        """v2.11: the code file defines `name` at top level (a def, class or assignment), read from the source."""
        return bool((self.source or "").strip()) and name in _module_level_names(self.source)

    def apply_patch(self, patch_text: str) -> str | None:
        """Apply a patch to the current source. Returns error or None on success."""
        if not self.source.strip():
            return "Cannot patch empty file. Use a full ```py rewrite first."
        try:
            new_source = _apply_patch(self.source, patch_text)
        except PatchError as e:
            return f"Patch failed: {e}"
        safe, hits = is_safe(new_source)
        if not safe:
            return safety_error(hits, 5, prefix="Safety check failed after patch")
        return self._try_source(new_source)

    def _compile(self) -> str | None:
        """Compile and exec the source into namespace. Returns error or None."""
        namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS, **preimported_namespace()}
        try:
            exec(self.source, namespace)
        except Exception as exc:
            self.compile_error = str(exc)
            self.namespace = None
            return f"Compilation/exec error: {exc!r}"
        self.namespace = namespace
        self.compile_error = None
        return None

    def inject_state(self, states: np.ndarray, actions_history: list[str]) -> None:
        """Inject runtime state into the namespace so code can reference it."""
        if self.namespace is not None:
            self.namespace["states"] = states
            self.namespace["actions_history"] = actions_history
            self.namespace["current_board"] = states[-1] if len(states) > 0 else None

    def get_next_state_fn(self) -> Callable[..., Any] | None:
        if self.namespace is None:
            return None
        fn = self.namespace.get("next_state")
        return fn if callable(fn) else None

    def get_callable(self, name: str) -> Callable[..., Any] | None:
        if self.namespace is None:
            return None
        fn = self.namespace.get(name)
        return fn if callable(fn) else None


class PatchError(Exception):
    pass


def _no_match_text(lines: list[str], context_lines: list[str], max_regions: int = 2, max_lines: int = 12) -> str:
    """Patch error for a hunk whose context matches nowhere: the most similar regions of the current
    file with line numbers, and the first line that differs."""
    import difflib
    n = len(context_lines)
    head = ("No match for the hunk's context (lines with ' ' and '-' prefixes must match the current file exactly, "
            "including indentation).")
    if n == 0 or not lines:
        return head
    target = "\n".join(context_lines)
    rough = []
    for i in range(0, max(1, len(lines) - n + 1)):
        window = "\n".join(lines[i:i + n])
        rough.append((difflib.SequenceMatcher(None, window, target, autojunk=False).quick_ratio(), i))
    rough.sort(reverse=True)
    scored = []
    for _, i in rough[:25]:
        window = "\n".join(lines[i:i + n])
        scored.append((difflib.SequenceMatcher(None, window, target, autojunk=False).ratio(), i))
    scored.sort(reverse=True)
    picked: list[tuple[float, int]] = []
    for ratio, i in scored:
        if all(abs(i - j) >= max(1, n) for _, j in picked):
            picked.append((ratio, i))
        if len(picked) >= max_regions:
            break
    out = [head]
    for ratio, i in picked:
        end = min(len(lines), i + n)
        out.append(f"Most similar region: lines {i + 1}-{end} (similarity {ratio:.2f}):")
        for k in range(i, min(end, i + max_lines)):
            out.append(f"  {k + 1:4d}| {lines[k]}")
        if end - i > max_lines:
            out.append(f"  ... {end - i - max_lines} more line(s)")
        for j in range(n):
            if i + j >= len(lines) or lines[i + j] != context_lines[j]:
                have = lines[i + j] if i + j < len(lines) else "<end of file>"
                out.append(f"  first difference at line {i + j + 1}: file has {have!r}, the patch expects {context_lines[j]!r}")
                break
    out.append("Copy the context lines exactly from the current file (run: print(CODE) shows it), or use py for a full rewrite.")
    return "\n".join(out)


def _apply_patch(source: str, patch_text: str) -> str:
    """Apply a simplified unified-diff patch (no filenames, no line numbers).

    Format: hunks separated by @@ lines. Each hunk line starts with:
      ' ' (space) = context line (must match)
      '-' = remove this line
      '+' = add this line
    """
    lines = source.split("\n")
    hunks = _parse_patch_hunks(patch_text)
    if not hunks:
        raise PatchError("No valid hunks found in patch")

    for hunk in reversed(hunks):
        lines = _apply_hunk(lines, hunk)
    return "\n".join(lines)


def _parse_patch_hunks(patch_text: str) -> list[list[tuple[str, str]]]:
    """Parse patch into list of hunks. Each hunk is list of (prefix, content)."""
    hunks: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] | None = None

    for raw_line in patch_text.split("\n"):
        if raw_line.strip() == "@@" or raw_line.startswith("@@"):
            if current is not None and current:
                hunks.append(current)
            current = []
            continue
        if current is None:
            continue
        if not raw_line:
            current.append((" ", ""))
            continue
        prefix = raw_line[0]
        content = raw_line[1:] if len(raw_line) > 1 else ""
        if prefix in (" ", "-", "+"):
            current.append((prefix, content))
        else:
            current.append((" ", raw_line))

    if current is not None and current:
        hunks.append(current)
    return hunks


def _match_positions(lines: list[str], context_lines: list[str]) -> list[int]:
    out: list[int] = []
    for i in range(len(lines) - len(context_lines) + 1):
        if all(lines[i + j] == ctx for j, ctx in enumerate(context_lines)):
            out.append(i)
    return out


def _apply_hunk(lines: list[str], hunk: list[tuple[str, str]]) -> list[str]:
    """Apply a single hunk to lines. Raises PatchError if no unique match."""
    context_lines = [content for prefix, content in hunk if prefix in (" ", "-")]
    match_positions = _match_positions(lines, context_lines)

    if not match_positions:
        # Blank context lines at the very start/end of a hunk are usually an artifact of the patch text's
        # trailing newline; retry without them before reporting a failure.
        trimmed = list(hunk)
        while trimmed and trimmed[-1] == (" ", ""):
            trimmed.pop()
        while trimmed and trimmed[0] == (" ", ""):
            trimmed.pop(0)
        if trimmed and len(trimmed) != len(hunk):
            t_ctx = [content for prefix, content in trimmed if prefix in (" ", "-")]
            t_pos = _match_positions(lines, t_ctx)
            if len(t_pos) == 1:
                hunk, context_lines, match_positions = trimmed, t_ctx, t_pos

    if len(match_positions) == 0:
        raise PatchError(_no_match_text(lines, context_lines))
    if len(match_positions) > 1:
        where = ", ".join(str(p + 1) for p in match_positions[:12])
        raise PatchError(
            f"Multiple matches ({len(match_positions)}): the hunk's context matches starting at lines {where}. "
            "Add more unchanged context lines so the hunk matches exactly once.")

    pos = match_positions[0]
    new_lines = lines[:pos]
    for prefix, content in hunk:
        if prefix == " ":
            new_lines.append(content)
        elif prefix == "+":
            new_lines.append(content)
    skip = len(context_lines)
    new_lines.extend(lines[pos + skip:])
    return new_lines


# ---------------------------------------------------------------- auto-kept functions (v2.3)

AUTO_KEPT_MARKER = ("# ==== auto-kept functions: top-level defs from run calls, added by the harness "
                    "(a later definition with the same name replaces the old one) ====")


def _def_span(node: ast.FunctionDef) -> tuple[int, int]:
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return start, int(node.end_lineno or node.lineno)


def top_level_functions(code: str, *, defined_functions: dict[str, int] | None = None) -> list[tuple[str, str]]:
    """Top-level defs in order, optionally restricted to worker-confirmed name/line pairs."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    lines = code.split("\n")
    out: list[tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            start, end = _def_span(node)
            if defined_functions is not None and defined_functions.get(node.name) != start:
                continue
            out.append((node.name, "\n".join(lines[start - 1:end])))
    return out


def merge_functions(source: str, funcs: list[tuple[str, str]]) -> str:
    """Put each function into `source`: replace the existing top-level definition with that name in place,
    otherwise append it to the auto-kept section at the end of the file."""
    for name, fsrc in funcs:
        replaced = False
        if source.strip():
            try:
                tree = ast.parse(source)
            except SyntaxError:
                tree = None
            if tree is not None:
                defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
                if defs:
                    lines = source.split("\n")
                    # remove every earlier duplicate, replace the last (effective) one
                    for n in reversed(defs[:-1]):
                        s0, e0 = _def_span(n)
                        del lines[s0 - 1:e0]
                    src2 = "\n".join(lines)
                    tree2 = ast.parse(src2)
                    last = [n for n in tree2.body if isinstance(n, ast.FunctionDef) and n.name == name][-1]
                    s1, e1 = _def_span(last)
                    lines2 = src2.split("\n")
                    lines2[s1 - 1:e1] = fsrc.split("\n")
                    source = "\n".join(lines2)
                    replaced = True
        if not replaced:
            if AUTO_KEPT_MARKER not in source:
                source = (source.rstrip("\n") + "\n\n\n" if source.strip() else "") + AUTO_KEPT_MARKER + "\n"
            source = source.rstrip("\n") + "\n\n" + fsrc + "\n"
    return source


RUN_NAMESPACE_NAMES = {   # v2.11: the helpers live in lib
    "np", "states", "current_board", "actions_history", "anim_frames", "get_animation", "history", "past_levels", "CODE",
    "indicator_rects", "act", "last_act_result", "store", "print", "lib",
    "notes", "data", "render", "extract_components", "find", "frame_diff", "anim_diff", "show", "describe", "propose_actions", "COLOR_NAMES",
}
MINIMAL_RESERVED_NAMES = frozenset(
    (RUN_NAMESPACE_NAMES - {"lib", "data", "indicator_rects"})
    | {"mark_remaining_actions_indicator", "indicator", "board_key", "click_candidates"}
    | set(SAFE_BUILTINS) | set(PREIMPORTED_MODULES) | {"__builtins__"}
)
# v2.10 helper names that are lib functions now (hint in the "not kept" message)
OLD_HELPER_HINTS = {
    "objects": "lib.objects", "frame_diff": "lib.frame_diff", "anim_diff": "lib.anim_diff", "obj": "lib.objects",
    "shifted_object": "lib.moved", "rotated_object": "lib.rotated", "mirrored_object": "lib.mirrored",
    "draw_objects": "lib.draw", "erase_objects": "lib.erase", "move_object": "lib.move_on_board",
    "indicator_rect": "indicator_rects",
}


def _module_level_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        else:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    names.add(sub.id)
    return names


def _free_names(func_src: str) -> set[str]:
    """Names a function reads that it does not define itself (parameters, locals, imports, comprehension vars)."""
    try:
        tree = ast.parse(func_src)
    except SyntaxError:
        return set()
    loaded: set[str] = set()
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):   # v2.10: lambda parameters are bound names
            a = node.args
            for arg in a.args + a.kwonlyargs + a.posonlyargs + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else []):
                bound.add(arg.arg)
        if isinstance(node, ast.FunctionDef):
            bound.add(node.name)
            a = node.args
            for arg in a.args + a.kwonlyargs + a.posonlyargs + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else []):
                bound.add(arg.arg)
        elif isinstance(node, ast.Name):
            (loaded if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                bound.add((al.asname or al.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return loaded - bound


def _reserved_helper_rejections(funcs, reserved_names):
    """Also reject dependants, whose meaning would change when a collision is dropped."""
    rejected = {n: "name is reserved for a harness API or Python built-in; rename it"
                for n, _ in funcs if n in reserved_names}
    while True:
        added = False
        for name, source in funcs:
            if name in rejected:
                continue
            dependencies = sorted(_free_names(source) & rejected.keys())
            if dependencies:
                rejected[name] = ("depends on rejected helper(s) " + ", ".join(dependencies)
                                  + "; rename them and update this helper")
                added = True
        if not added:
            return rejected


def strip_reserved_helpers(source: str, reserved_names) -> tuple[str, list[str]]:
    """Ignore old conflicting definitions before loading, preserving source line numbers."""
    rejected = _reserved_helper_rejections(top_level_functions(source), reserved_names)
    if not rejected:
        return source, []
    lines = source.splitlines(keepends=True)
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in rejected:
            start, end = _def_span(node)
            lines[start - 1:end] = ["\n"] * (end - start + 1)
    return "".join(lines), [f"{n} ({reason})" for n, reason in rejected.items()]


def keep_functions_from_run(code_file: "VirtualCodeFile", code: str, *,
                            reserved_names=frozenset(), defined_functions=None) -> tuple[list[str], str | None]:
    """Auto-keep the top-level functions defined by a run snippet in the code file.
    Returns (names whose definition changed, error or None). On a compile error nothing changes.
    Optional reserved names reject collisions and their dependant helpers without changing local run execution.
    After RESET, defined_functions restricts retention to definitions confirmed by the worker before stopping.
    A function that reads a name defined only in the snippet (not in the code file, the run namespace or the
    builtins) is NOT kept, and the error says which name is missing."""
    funcs = top_level_functions(code, defined_functions=defined_functions)
    if not funcs:
        return [], None
    collisions = _reserved_helper_rejections(funcs, reserved_names)
    funcs = [(n, f) for n, f in funcs if n not in collisions]
    rejected = [f"{n} ({reason})" for n, reason in collisions.items()]
    def rejection_text():
        if not rejected:
            return None
        text = "not kept: " + "; ".join(rejected)
        if collisions:
            text += (". This only prevents persistence: conflicting definitions can still shadow names "
                     "within this run; later runs use the provided built-ins.")
        return text
    old = code_file.source or ""
    existing = dict(top_level_functions(old))
    changed = [(n, f) for n, f in funcs if existing.get(n) != f]
    # keep the last definition when a snippet defines the same name twice
    dedup: dict[str, str] = {}
    for n, f in changed:
        dedup[n] = f
    changed = list(dedup.items())
    if not changed:
        return [], rejection_text()
    # dependency check: what will be defined after the merge?
    available = _module_level_names(old) | {n for n, _ in changed} | RUN_NAMESPACE_NAMES | set(SAFE_BUILTINS) | set(ALLOWED_IMPORT_MODULES)
    available |= {"True", "False", "None", "__builtins__"}
    kept_changed: list[tuple[str, str]] = []
    for n, f in changed:
        missing = sorted(x for x in _free_names(f) if x not in available)
        if missing:
            hints = [f"{x} is {OLD_HELPER_HINTS[x]} now" for x in missing if x in OLD_HELPER_HINTS]
            rejected.append(f"{n} (uses {', '.join(missing)}, not defined in your code file; pass it as a parameter or add it "
                            "with py/patch" + (f"; {', '.join(hints)}" if hints else "") + ")")
        else:
            kept_changed.append((n, f))
    changed = kept_changed
    if not changed:
        return [], rejection_text()
    new = merge_functions(old, changed)
    err = code_file.update_full(new)
    if err:   # update_full changes nothing when loading fails: the previous file stays active
        return [], f"not kept (the code file would not compile with them: {err})" + ("; " + rejection_text() if rejected else "")
    return [n for n, _ in changed], rejection_text()
