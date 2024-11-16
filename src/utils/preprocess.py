import ast
import tree_sitter_java as tsj
import tree_sitter_c as tsc
import tree_sitter_rust as tsr
from tree_sitter import Language, Parser
import tree_sitter


def clean_python_code(code: str) -> str:
    """
    Clean the given python code snippet:
    - remove empty lines
    - remove comments
    """

    def remove_docstring_recursively(node: ast.AST):
        if isinstance(
            node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef, ast.Module)
        ):
            if node.body and isinstance(node.body[0], ast.Expr):
                first_child = node.body[0].value
                if (
                    isinstance(first_child, ast.Str)
                    or isinstance(first_child, ast.Constant)
                    and isinstance(first_child.value, str)
                ):
                    # has docstring, remove it (first node in body)
                    node.body = node.body[1:]
        if hasattr(node, "body"):
            for child in node.body:
                remove_docstring_recursively(child)

    tree = ast.parse(code)
    remove_docstring_recursively(tree)
    return ast.unparse(tree)


def split_python_functions(code: str) -> list[(str, bool)]:
    fns = []

    # nested functions are not considered
    class_decl_stack = []

    def extract_function_recursively(node: ast.AST):
        nonlocal class_decl_stack
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            fns.append((node, len(class_decl_stack) > 0))
        else:
            if hasattr(node, "body"):
                for child in node.body:
                    if isinstance(child, ast.ClassDef):
                        class_decl_stack.append(child)
                    extract_function_recursively(child)
                    if isinstance(child, ast.ClassDef):
                        class_decl_stack.pop()

    tree = ast.parse(code)
    extract_function_recursively(tree)
    return list(map(lambda x: (ast.unparse(x[0]), x[1]), fns))


class TestCleanCodePatternBased:
    def test_clean_empty_lines(self):
        code = """
import pathlib 

pathlib.Path(".")
        """
        cleaned_code = clean_python_code(code)
        assert cleaned_code == "import pathlib\npathlib.Path('.')"

    def test_clean_line_comment(self):
        code = """
# comment 1
import pathlib # comment 2
# comment 3
"""
        cleaned_code = clean_python_code(code)
        assert cleaned_code == "import pathlib"

    def test_clean_block_comment(self):
        code = """
def foo():
    \"""
    docstring
    \"""
    pass
"""
        cleaned_code = clean_python_code(code)
        print(cleaned_code)
        assert cleaned_code == "def foo():\n    pass"


class TestSplitFunctions:
    def test_split_top_level_functions(self):
        code = """
def foo():
    print(123)

def bar():
    print(456)
"""
        fns = split_python_functions(code)
        assert len(fns) == 2
        assert fns[0][0] == "def foo():\n    print(123)"
        assert fns[0][1] is False
        assert fns[1][0] == "def bar():\n    print(456)"
        assert fns[1][1] is False

    def test_split_class_functions(self):
        code = """
class C:
    @staticmethod
    def foo():
        print(123)

    def bar(self):
        print(456)
"""
        fns = split_python_functions(code)
        assert len(fns) == 2
        assert fns[0][0] == "@staticmethod\ndef foo():\n    print(123)"
        assert fns[0][1] is True
        assert fns[1][0] == "def bar(self):\n    print(456)"
        assert fns[1][1] is True


JAVA_LANG = Language(tsj.language())


def split_java_classes(code: str) -> list[str]:
    # parse using tree-sitter
    parser = Parser(JAVA_LANG)
    ast = parser.parse(code.encode())

    classes = []

    def find_top_level_class(node):
        if node.type == "class_declaration":
            classes.append(node)
            return
        for child in node.children:
            find_top_level_class(child)

    find_top_level_class(ast.root_node)

    return list(
        map(
            lambda x: code.encode("utf-8")[x.start_byte : x.end_byte].decode("utf-8"),
            classes,
        )
    )


class TestSplitJavaClasses:
    def test_split_single_class(self):
        code = """
public class A {
    public void foo() {}
}
public class B {}
"""
        classes = split_java_classes(code)
        assert len(classes) == 2
        assert classes[0] == "public class A {\n    public void foo() {}\n}"


def count_methods_in_java_class(code: str) -> int:
    parser = Parser(JAVA_LANG)
    ast = parser.parse(code.encode())

    def count_methods(node):
        if node.type == "method_declaration":
            return 1
        return sum(count_methods(child) for child in node.children)

    return count_methods(ast.root_node)


class TestCountMethodsInJavaClass:
    def test_count_methods(self):
        code = """
class A {
    void foo() {}
    void bar() {}
}
"""
        assert count_methods_in_java_class(code) == 2


def get_all_methods_in_java_class(code: str) -> list[str]:
    parser = Parser(JAVA_LANG)
    ast = parser.parse(code.encode())

    def find_methods(node):
        if node.type == "method_declaration":
            return [code[node.start_byte : node.end_byte]]
        return sum((find_methods(child) for child in node.children), [])

    return find_methods(ast.root_node)


def find_all_ts_nodes_by_type(node, type_name):
    if node.type == type_name:
        yield node
    for child in node.children:
        yield from find_all_ts_nodes_by_type(child, type_name)


def count_statements_in_java_method(code: str) -> int:
    parser = Parser(JAVA_LANG)
    ast = parser.parse(code.encode())

    # find all method_declaration nodes
    method_nodes: list[tree_sitter.Node] = list(
        find_all_ts_nodes_by_type(ast.root_node, "method_declaration")
    )
    assert len(method_nodes) == 1
    method = method_nodes[0]
    body = method.child_by_field_name("body")
    body = list(filter(lambda n: n.type not in ["{", "}"], body.children))
    return len(body)


class TestCountStatementsInJavaMethod:
    def test_count_statements(self):
        code = """
void foo() {
    int a = 1;
    int b = 2;
}
"""
        assert count_statements_in_java_method(code) == 2


C_LANG = Language(tsc.language())


def split_c_functions(code: str) -> list[str]:
    # parse using tree-sitter
    parser = Parser(C_LANG)
    ast = parser.parse(code.encode())

    fns = []

    def find_top_level_function(node):
        if node.type == "function_definition":
            fns.append(node)
            return
        for child in node.children:
            find_top_level_function(child)

    find_top_level_function(ast.root_node)

    return list(
        map(
            lambda x: code.encode("utf-8")[x.start_byte : x.end_byte].decode("utf-8"),
            fns,
        )
    )


class TestSplitCFunctions:
    def test_split_single_function(self):
        code = """
void foo() {}
void bar() {}
"""
        fns = split_c_functions(code)
        assert len(fns) == 2
        assert fns[0] == "void foo() {}"


def count_statements_in_c_function(code: str) -> int:
    parser = Parser(C_LANG)
    ast = parser.parse(code.encode())

    # find all function_definition nodes
    function_nodes: list[tree_sitter.Node] = list(
        find_all_ts_nodes_by_type(ast.root_node, "function_definition")
    )
    assert len(function_nodes) == 1
    function = function_nodes[0]
    body = function.child_by_field_name("body")
    body = list(filter(lambda n: n.type not in ["{", "}"], body.children))
    return len(body)


class TestCountStatementsInCFunction:
    def test_count_statements(self):
        code = """
void foo() {
    int a = 1;
    int b = 2;
}
"""
        assert count_statements_in_c_function(code) == 2

    def test_count_statements_empty(self):
        code = """
void foo() {}
"""
        assert count_statements_in_c_function(code) == 0


RUST_LANG = Language(tsr.language())


def split_rust_functions(code: str) -> list[str]:
    # parse using tree-sitter
    parser = Parser(RUST_LANG)
    ast = parser.parse(code.encode())

    fns = []

    def find_top_level_function(node):
        if node.type == "impl_item":
            # ignore impl blocks
            return
        if node.type == "function_item":
            fns.append(node)
            return
        for child in node.children:
            find_top_level_function(child)

    find_top_level_function(ast.root_node)

    return list(
        map(
            lambda x: code.encode("utf-8")[x.start_byte : x.end_byte].decode("utf-8"),
            fns,
        )
    )


class TestSplitRustFunctions:
    def test_split_single_function(self):
        code = """
struct S;
impl S {
    fn foo() {}
    fn bar() {}
}
fn foo() {}
"""
        fns = split_rust_functions(code)
        assert len(fns) == 1
        assert fns[0] == "fn foo() {}"


def count_statements_in_rust_function(code: str) -> int:
    parser = Parser(RUST_LANG)
    ast = parser.parse(code.encode())

    # find all function_item nodes
    function_nodes: list[tree_sitter.Node] = list(
        find_all_ts_nodes_by_type(ast.root_node, "function_item")
    )
    assert len(function_nodes) == 1
    function = function_nodes[0]
    body = function.child_by_field_name("body")
    body = list(filter(lambda n: n.type not in ["{", "}"], body.children))
    return len(body)


class TestCountStatementsInRustFunction:
    def test_count_statements(self):
        code = """
fn foo() {
    let a = 1;
    let b = 2;
}
"""
        assert count_statements_in_rust_function(code) == 2

    def test_count_statements_empty(self):
        code = """
fn foo() {}
"""
        assert count_statements_in_rust_function(code) == 0
