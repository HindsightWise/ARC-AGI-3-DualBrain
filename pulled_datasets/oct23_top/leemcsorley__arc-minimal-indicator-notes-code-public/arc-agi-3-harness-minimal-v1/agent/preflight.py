"""Check a staged harness before GPU setup, without importing game/model dependencies.

Usage: python agent/preflight.py [path/to/agent]
Checks syntax and unresolved module globals, including names used inside functions.
This catches missing imports; it is not a substitute for running the game loop.
"""
from __future__ import annotations

import builtins
from pathlib import Path
import symtable
import sys


def check_package(package_dir: Path) -> list[str]:
    errors: list[str] = []
    modules = sorted(package_dir.glob("*.py"))
    if not (package_dir / "agent.py").is_file():
        return [f"{package_dir}: missing agent.py"]
    implicit = set(dir(builtins)) | {
        "__file__", "__name__", "__package__", "__spec__", "__loader__", "__cached__", "__builtins__",
    }
    for path in modules:
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            table = symtable.symtable(source, str(path), "exec")
        except (OSError, UnicodeError, SyntaxError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        defined = {s.get_name() for s in table.get_symbols()
                   if s.is_assigned() or s.is_imported() or s.is_namespace()} | implicit

        def check_scope(scope):
            for symbol in scope.get_symbols():
                if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in defined:
                    errors.append(f"{path.name}:{scope.get_lineno()} ({scope.get_name()}): "
                                  f"undefined global {symbol.get_name()!r}")
            for child in scope.get_children():
                check_scope(child)

        check_scope(table)
    return errors


def main() -> int:
    package_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    errors = check_package(package_dir)
    if errors:
        print("Harness preflight FAILED before GPU setup:\n" + "\n".join(errors), file=sys.stderr)
        return 1
    print(f"Harness preflight passed: {len(list(package_dir.glob('*.py')))} modules; syntax and global names checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
