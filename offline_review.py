"""Python diagnostics based on syntax, without executing reviewed source."""
import ast
from collections import Counter


def scan_python(code: str) -> dict[str, list[str]]:
    findings = {"code_debug": [], "code_improvements": [], "performance": [], "security": []}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return findings

    def add(node: ast.AST, issue: str, fix: str, *, category: str = "code_debug") -> None:
        findings[category].append(f"Line {getattr(node, 'lineno', 1)}: {issue}")
        findings["code_improvements"].append(f"Line {getattr(node, 'lineno', 1)}: {fix}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = [*node.args.defaults, *[item for item in node.args.kw_defaults if item is not None]]
            for default in defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    add(default, "Mutable default argument is shared across calls.",
                        "Use None as the default and create a fresh list, dict, or set inside the function.")
            parameters = {arg.arg for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]}
            guarded = set()
            for branch in ast.walk(node):
                if not isinstance(branch, (ast.If, ast.IfExp)):
                    continue
                test = branch.test
                for part in ast.walk(test):
                    if isinstance(part, ast.Name) and isinstance(test, (ast.Name, ast.UnaryOp, ast.BoolOp)):
                        guarded.add(part.id)
                    if (isinstance(part, ast.Call) and isinstance(part.func, ast.Name) and part.func.id == "len"
                            and part.args and isinstance(part.args[0], ast.Name)):
                        guarded.add(part.args[0].id)
            for access in ast.walk(node):
                if (isinstance(access, ast.Subscript) and isinstance(access.value, ast.Name)
                        and access.value.id in parameters - guarded and isinstance(access.slice, ast.Constant)
                        and access.slice.value == 0):
                    add(access, f"Access to {access.value.id}[0] can raise IndexError for an empty sequence.",
                        f"Check for empty input before indexing {access.value.id}; return or raise explicitly.")
            for statement in ast.walk(node):
                if not isinstance(statement, ast.Return):
                    continue
                calls = Counter(part.func.id for part in ast.walk(statement)
                                if isinstance(part, ast.Call) and isinstance(part.func, ast.Name))
                if calls[node.name] > 1:
                    add(statement, "Repeated recursive calls can produce exponential work.",
                        "Use memoization or an iterative dynamic-programming algorithm.", category="performance")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                add(node, "Division or modulo by zero raises ZeroDivisionError.",
                    "Validate the denominator and handle zero before dividing.")
        elif isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for index, operator in enumerate(node.ops):
                pair = operands[index:index + 2]
                if isinstance(operator, (ast.Is, ast.IsNot)) and any(
                    isinstance(value, ast.Constant) and value.value is not None and not isinstance(value.value, bool)
                    for value in pair
                ):
                    add(node, "Identity comparison with a literal does not reliably compare values.",
                        "Use == or != for value comparison; reserve is for None or singleton objects.")
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id == "sorted":
                add(node, "The return value of sorted() is discarded; the original collection is unchanged.",
                    "Assign or return sorted(items), or use list.sort() for an in-place sort.")
        elif isinstance(node, ast.Call):
            call = node.func
            if isinstance(call, ast.Name) and call.id in {"eval", "exec"}:
                add(node, f"{call.id}() can execute untrusted input.",
                    "Avoid dynamic execution; use explicit parsing or a fixed dispatch table.", category="security")
            if (isinstance(call, ast.Attribute) and isinstance(call.value, ast.Name)
                    and call.value.id == "requests" and call.attr in {"get", "post", "put", "delete", "patch", "head"}):
                timeout = next((keyword.value for keyword in node.keywords if keyword.arg == "timeout"), None)
                if timeout is None or isinstance(timeout, ast.Constant) and timeout.value is None:
                    add(node, "HTTP request without a timeout can hang indefinitely.",
                        "Pass a finite timeout to the requests call.", category="security")
    return {key: list(dict.fromkeys(values)) for key, values in findings.items()}
