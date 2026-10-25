"""Preserve all printed evidence and one Python namespace in PythonAstREPLTool."""
import ast
from contextlib import redirect_stdout
from io import StringIO
from langchain_experimental.tools.python.tool import PythonAstREPLTool, sanitize_input


class CompatiblePythonAstREPLTool(PythonAstREPLTool):
    def _run(self, query, run_manager=None):
        buffer = StringIO()
        try:
            query = sanitize_input(query) if self.sanitize_input else query
            tree = ast.parse(query)
            # Functions, comprehensions and later calls must see the same variables.
            namespace = self.locals
            self.globals = namespace
            with redirect_stdout(buffer):
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]), '<repl>', 'exec'), namespace, namespace)
                    result = eval(compile(ast.Expression(tree.body[-1].value), '<repl>', 'eval'), namespace, namespace)
                else:
                    exec(compile(tree, '<repl>', 'exec'), namespace, namespace)
                    result = None
            printed = buffer.getvalue()
            if result is None:
                return printed
            return printed + str(result) if printed else result
        except Exception as exc:
            return buffer.getvalue() + f'{type(exc).__name__}: {exc}'


def apply_repl_compat(executor):
    """Keep the original tool name/schema/description and DataFrame instances."""
    tools = []
    for tool in executor.tools:
        if isinstance(tool, PythonAstREPLTool):
            namespace = dict(tool.globals)
            namespace.update(tool.locals)
            replacement = CompatiblePythonAstREPLTool(
                name=tool.name, description=tool.description, args_schema=tool.args_schema,
                sanitize_input=tool.sanitize_input,
            )
            replacement.locals = namespace
            replacement.globals = namespace
            tools.append(replacement)
        else:
            tools.append(tool)
    executor.tools = tools
    return executor
