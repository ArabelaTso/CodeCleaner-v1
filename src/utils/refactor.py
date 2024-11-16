from abc import ABC, abstractmethod
import time
# from distutils.command.install import value

import numpy as np
from collections import defaultdict
import ast

# from sympy import false
from wordhoard import Synonyms
import re
import copy


class RefactorOperator(ABC):
    @abstractmethod
    def refactor(
        self, root: ast.AST, rand: bool = False, max_count: int = 1
    ) -> tuple[ast.AST, int]:
        """
        Refactor the given AST in-place.
        CAUTION: The original given AST will be polluted.
        @param root: the root node of the AST
        @param rand: whether to apply the refactor randomly. There usually may be multiple places to apply the refactor. If rand=True, the refactor will be applied to at most max_count places, otherwise, all the applicable places will be refactored.
        @param max_count: the maximum number of refactor to apply (only applicable when rand=True)
        @return: the refactored ast and the number of times that the refactor is applied
        """
        pass


class GroupRefactor(RefactorOperator):
    """
    Apply a group of refactors to the given AST.
    """

    def __init__(self, *operators: list[RefactorOperator]) -> None:
        self.operators = operators

    def refactor(
        self, root: ast.AST, rand: bool = False, max_count: int = 1
    ) -> tuple[ast.AST, int]:
        count = 0
        if rand:
            new_refector = True
            while count < max_count and new_refector:
                new_refector = False
                for operator in self.operators:
                    # mc = np.random.randint(0, max_count - count + 1)
                    mc = 1
                    if mc > 0:
                        root, c = operator.refactor(root, rand=True, max_count=mc)
                    else:
                        c = 0
                    count += c
                    if c > 0:
                        new_refector = True
                    if count >= max_count:
                        break
            return root, count
        else:
            for operator in self.operators:
                root, c = operator.refactor(root, rand=False)
                count += c
            return root, count


class IfBranchFliper(RefactorOperator):
    def refactor(
        self, root: ast.AST, rand: bool = False, max_count: int = 1
    ) -> tuple[ast.AST, int]:
        assert not rand or rand and max_count > 0

        def collect_if(node: ast.AST) -> list[ast.If]:
            ifs = []
            if isinstance(node, ast.If):
                ifs.append(node)
            for child in ast.iter_child_nodes(node):
                ifs += collect_if(child)
            return ifs

        ifs = collect_if(root)

        if rand:
            selected = (
                np.random.choice(ifs, max_count, replace=False) if len(ifs) > 0 else []
            )
        else:
            selected = ifs

        for node in selected:
            # flip the if-else branch
            cond = node.test
            true_branch = node.body
            false_branch = node.orelse if len(node.orelse) > 0 else [ast.Pass()]
            not_cond = ast.UnaryOp(op=ast.Not(), operand=cond)
            node.test = not_cond
            node.body = false_branch
            node.orelse = true_branch

        return root, len(selected)


class TestIfBranchFliper:
    def test_simple(self):
        code = """
if not True:
    assert True
else: 
    pass
"""
        root = ast.parse(code)
        root, count = IfBranchFliper().refactor(root)
        assert count == 1
        assert ast.unparse(root) == "if not not True:\n    pass\nelse:\n    assert True"

    def test_one_branch(self):
        code = """
if Ture:
    assert True
"""
        root = ast.parse(code)
        root, count = IfBranchFliper().refactor(root)
        assert count == 1
        assert ast.unparse(root) == "if not Ture:\n    pass\nelse:\n    assert True"

    def test_recursive(self):
        code = """
if True:
    if False:
        assert True
    else:
        assert False
else:
    pass
"""
        root = ast.parse(code)
        root, count = IfBranchFliper().refactor(root)
        assert count == 2
        assert (
            ast.unparse(root)
            == "if not True:\n    pass\nelif not False:\n    assert False\nelse:\n    assert True"
        )

    def test_elif(self):
        code = """
if x > 0:
    assert True
elif x < 0:
    assert False
else:
    pass
"""
        root = ast.parse(code)
        root, count = IfBranchFliper().refactor(root)
        assert count == 2
        assert (
            ast.unparse(root)
            == "if not x > 0:\n    if not x < 0:\n        pass\n    else:\n        assert False\nelse:\n    assert True"
        )

    def test_random(self):
        code = """
if x > 0:
    assert True
if x < 0:
    assert False
"""
        root = ast.parse(code)
        root, count = IfBranchFliper().refactor(root, rand=True, max_count=1)
        assert count == 1
        assert (
            ast.unparse(root)
            == "if x > 0:\n    assert True\nif not x < 0:\n    pass\nelse:\n    assert False"
            or ast.unparse(root)
            == "if not x > 0:\n    pass\nelse:\n    assert True\nif x < 0:\n    assert False"
        )


class While2For(RefactorOperator):
    def __init__(self) -> None:
        super().__init__()
        self.inifinite_generator = ast.parse("iter(lambda: 0, 1)").body[0].value

    def refactor(self, root: ast.AST, rand: bool = False, max_count: int = 1) -> int:
        assert not rand or rand and max_count > 0

        whiles = []
        # breadth-first traversal
        queue = [(root, None, None, None)]
        while len(queue) > 0:
            node, parent, field, index = queue.pop(0)
            if isinstance(node, ast.While):
                whiles.append((parent, field, index, node))
            for field, list_child in ast.iter_fields(node):
                if not isinstance(list_child, list):
                    continue
                for i, child in enumerate(list_child):
                    if not isinstance(child, ast.AST):
                        continue
                    queue.append((child, node, field, i))
        whiles = list(reversed(whiles))
        # the while nodes are sorted by its depth in AST

        if rand:
            selected_idxs = (
                np.random.choice(len(whiles), max_count, replace=False)
                if len(whiles) > 0
                else []
            )
            selected = [whiles[idx] for idx in selected_idxs]
        else:
            selected = whiles

        new_root = root
        for parent, field, index, node in selected:
            # convert while loop to for loops
            cond = node.test
            body = node.body
            orelse = node.orelse
            term_cond = ast.UnaryOp(op=ast.Not(), operand=cond)
            term_if = ast.If(test=term_cond, body=orelse + [ast.Break()], orelse=[])
            new_body = [term_if] + body
            for_node = ast.For(
                target=ast.Name(id="_", ctx=ast.Store()),
                iter=self.inifinite_generator,
                body=new_body,
                orelse=[],
                lineno=node.lineno,
            )
            if parent is None:
                new_root = for_node
            else:
                parent.__getattribute__(field).__setitem__(index, for_node)

        return new_root, len(selected)


class TestWhile2For:
    infinite_generator = "iter(lambda: 0, 1)"

    def test_simple(self):
        code = """
while True:
    pass
"""
        root = ast.parse(code)
        root, count = While2For().refactor(root)
        assert count == 1
        assert (
            ast.unparse(root)
            == f"for _ in {self.infinite_generator}:\n    if not True:\n        break\n    pass"
        )

    def test_else(self):
        code = """
while True:
    pass
else:
    assert True
"""
        root = ast.parse(code)
        root, count = While2For().refactor(root)
        assert count == 1
        assert (
            ast.unparse(root)
            == f"for _ in {self.infinite_generator}:\n    if not True:\n        assert True\n        break\n    pass"
        )

    def test_recursive(self):
        code = """
while True:
    while False:
        assert False
    assert True
"""
        root = ast.parse(code)
        root, count = While2For().refactor(root)
        assert count == 2
        assert (
            ast.unparse(root)
            == f"for _ in {self.infinite_generator}:\n    if not True:\n        break\n    for _ in {self.infinite_generator}:\n        if not False:\n            break\n        assert False\n    assert True"
        )


class For2While(RefactorOperator):
    def __init__(self) -> None:
        super().__init__()
        self.idx = 0

    def refactor(
        self, root: ast.AST, rand: bool = False, max_count: int = 1
    ) -> tuple[ast.AST, int]:
        assert not rand or rand and max_count > 0

        fors = []
        # breadth-first traversal
        queue = [(root, None, None, None)]
        while len(queue) > 0:
            node, parent, field, index = queue.pop(0)
            if isinstance(node, ast.For):
                fors.append((parent, field, index, node))
            for field, list_child in ast.iter_fields(node):
                if not isinstance(list_child, list):
                    continue
                for i, child in enumerate(list_child):
                    if not isinstance(child, ast.AST):
                        continue
                    queue.append((child, node, field, i))
        fors = list(reversed(fors))
        # the for nodes are sorted by its depth in AST

        if rand:
            selected_idxs = (
                np.random.choice(len(fors), max_count, replace=False)
                if len(fors) > 0
                else []
            )
            selected = [fors[idx] for idx in selected_idxs]
        else:
            selected = fors

        new_root = root
        for parent, field, index, node in selected:
            # convert for loop to while loops
            body = node.body
            orelse = node.orelse
            iter_call = node.iter
            iter_var = ast.Name(id=f"_iter{self.idx}", ctx=ast.Store())
            self.idx += 1
            iter_assign = ast.Assign(
                targets=[iter_var], value=iter_call, lineno=node.lineno
            )
            receivers = [node.target]
            iter_next = ast.Assign(
                targets=receivers,
                value=ast.Call(
                    func=ast.Name(id="next", ctx=ast.Load()),
                    args=[iter_var],
                    keywords=[],
                ),
                lineno=node.lineno,
            )
            try_catch = ast.Try(
                body=[iter_next],
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id="StopIteration"),
                        ctx=ast.Load(),
                        body=orelse + [ast.Break()],
                    )
                ],
                orelse=[],
                finalbody=[],
                lineno=node.lineno,
            )
            while_node = ast.While(
                test=ast.Constant(value=True),
                body=[try_catch] + body,
                orelse=[],
                lineno=node.lineno,
            )
            if parent is None:
                new_root = while_node
            else:
                parent.__getattribute__(field).__setitem__(index, while_node)
                parent.__getattribute__(field).insert(index, iter_assign)

        return new_root, len(selected)


class TestFor2While:
    def test_simple(self):
        code = """
for i in range(10):
    pass
"""
        root = ast.parse(code)
        root, count = For2While().refactor(root)
        assert count == 1
        assert (
            ast.unparse(root)
            == "_iter0 = range(10)\nwhile True:\n    try:\n        i = next(_iter0)\n    except StopIteration:\n        break\n    pass"
        )

    def test_tuple(self):
        code = """
for i, j in enumerate(range(10)):
    pass
"""
        root = ast.parse(code)
        root, count = For2While().refactor(root)
        assert count == 1
        assert (
            ast.unparse(root)
            == "_iter0 = enumerate(range(10))\nwhile True:\n    try:\n        i, j = next(_iter0)\n    except StopIteration:\n        break\n    pass"
        )


class FnVarargAppender(RefactorOperator):
    def refactor(
        self, root: ast.AST, rand: bool = False, max_count: int = 1
    ) -> tuple[ast.AST, int]:
        def collect_fns(node: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
            fns = []
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.args.vararg is None
            ):
                fns.append(node)
            for child in ast.iter_child_nodes(node):
                fns += collect_fns(child)
            return fns

        fns = collect_fns(root)
        selected = (
            np.random.choice(fns, max_count, replace=False)
            if rand and len(fns) > 0
            else fns
        )

        for fn in selected:
            fn.args.vararg = ast.arg(arg="args")

        return root, len(selected)


class TestVarargAppender:
    def test_simple(self):
        code = """
def foo():
    pass
"""
        root = ast.parse(code)
        root, count = FnVarargAppender().refactor(root)
        assert count == 1
        assert ast.unparse(root) == "def foo(*args):\n    pass"

    def test_multiple_fn(self):
        code = """
def foo():
    pass
def bar():
    pass
"""
        root = ast.parse(code)
        root, count = FnVarargAppender().refactor(root)
        assert count == 2
        assert (
            ast.unparse(root)
            == "def foo(*args):\n    pass\n\ndef bar(*args):\n    pass"
        )


class FnKwargAppender(RefactorOperator):
    def refactor(
        self, root: ast.AST, rand: bool = False, max_count: int = 1
    ) -> tuple[ast.AST, int]:
        def collect_fns(node: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
            fns = []
            if (
                isinstance(node, (ast.FunctionDef | ast.AsyncFunctionDef))
                and node.args.kwarg is None
            ):
                fns.append(node)
            for child in ast.iter_child_nodes(node):
                fns += collect_fns(child)
            return fns

        fns = collect_fns(root)
        selected = (
            np.random.choice(fns, max_count, replace=False)
            if rand and len(fns) > 0
            else fns
        )

        for fn in selected:
            fn.args.kwarg = ast.arg(arg="kwargs")

        return root, len(selected)


class TestKwargAppender:
    def test_simple(self):
        code = """
def foo():
    pass
"""
        root = ast.parse(code)
        root, count = FnKwargAppender().refactor(root)
        assert count == 1
        assert ast.unparse(root) == "def foo(**kwargs):\n    pass"


class FnDecorator(RefactorOperator):
    def __init__(
        self,
        *decorators: list[str | ast.expr],
    ) -> None:
        super().__init__()
        self.decorators = list(
            map(
                lambda d: ast.parse(f"{d}\ndef foo():\n    pass")
                .body[0]
                .decorator_list[0]
                if isinstance(d, str)
                else d,
                decorators,
            )
        )

    def refactor(
        self, root: ast.AST, rand: bool = False, max_count: int = 1
    ) -> tuple[ast.AST, int]:
        def collect_fns(node: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
            fns = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fns.append(node)
            for child in ast.iter_child_nodes(node):
                fns += collect_fns(child)
            return fns

        fns = collect_fns(root)
        selected = (
            np.random.choice(fns, max_count, replace=False)
            if rand and len(fns) > 0
            else fns
        )

        for fn in selected:
            for decorator in self.decorators:
                fn.decorator_list.append(decorator)

        return root, len(selected)


class TestFnDecorator:
    def test_simple(self):
        code = """
def foo():
    pass
"""
        root = ast.parse(code)
        root, count = FnDecorator("@timing").refactor(root)
        assert count == 1
        assert ast.unparse(root) == "@timing\ndef foo():\n    pass"

    def test_multiple(self):
        code = """
def foo():
    pass
"""
        root = ast.parse(code)
        root, count = FnDecorator("@timing", "@measure_memory_usage").refactor(root)
        assert count == 1
        assert (
            ast.unparse(root) == "@timing\n@measure_memory_usage\ndef foo():\n    pass"
        )


class VarRenamer(RefactorOperator):
    def refactor(
        self, root: ast.AST, rand: bool = False, max_count: int = 1
    ) -> tuple[ast.AST, int]:
        def collect_vars(node: ast.AST, startcollect=False) -> list[ast.Name]:
            if isinstance(node, ast.FunctionDef):
                startcollect = True
            vars = []
            if startcollect:
                if (isinstance(node, ast.Name) and node.id not in ['self', 'super', 'cls']) or \
                   (isinstance(node, ast.arg) and node.arg not in ['self', 'super', 'cls']):
                    vars.append(node)
            decorator_set = frozenset(node.decorator_list) if hasattr(node, 'decorator_list') else frozenset() ##
            for child in ast.iter_child_nodes(node):
                if child not in decorator_set: ##
                    vars += collect_vars(child, startcollect)
            return vars

        vars = collect_vars(root)
        # filter local vars
        name2vars = defaultdict(list)
        for var in vars:
            name2vars[(var.id if isinstance(var, ast.Name) else var.arg)].append(var)
        keys = list(name2vars.keys())
        for name in keys:
            vars = [x for x in name2vars[name] if isinstance(x, ast.Name)]
            # local vars are approximated as those that are assigned values
            if not any(map(lambda v: isinstance(v.ctx, ast.Store), vars)):
                del name2vars[name]

        if len(name2vars) == 0:
            return root, 0

        if rand:
            n = min(max_count, len(name2vars))
            names = set(name2vars.keys())
            selected2synonym = {}
            used = set(['yield'])
            while 0 <= len(selected2synonym) < n and len(names) > 0:
                selected = np.random.choice(list(names), 1, replace=False)[0]
                names.remove(selected)
                selected_words = selected.split('_')
                synonyms = []
                for i, word in enumerate(selected_words):
                    syns = Synonyms(word).find_synonyms()
                    print(f"fetched synonyms for {word}: {syns}")
                    time.sleep(3)
                    if syns is None or len(syns) == 0:
                        continue
                    for syn in syns:
                        if re.search(r"^[0-9]", syn):
                            continue
                        selected_words[i] = syn
                        synonym_name = '_'.join(selected_words)
                        synonyms.append(synonym_name)
                    break
                for synonym in synonyms:
                    if synonym in used:
                        continue
                    selected2synonym[selected] = synonym
                    break
        else:
            tmp = []
            for name in name2vars.keys():
                tmp.append((name, Synonyms(name).find_synonyms()))
                time.sleep(5)
            name2synonyms = filter(lambda it: it[1] is not None and len(it[1]) > 0, tmp)
            selected2synonym = {}
            used = set()
            for name, synonyms in name2synonyms:
                for synonym in synonyms:
                    if synonym in used:
                        continue
                    selected2synonym[name] = synonym

        for name, synonym in selected2synonym.items():
            vars = name2vars[name]
            words = synonym.split()
            if len(words) > 1:
                new_id = '_'.join(words)
            else:
                new_id = synonym
            new_id = re.sub(r'[0-9\-\(\)\[\]\{\}\']', '_', new_id)
            print(f"renaming {name} => {new_id}")
            for var in vars:
                if isinstance(var, ast.Name):
                    var.id = new_id
                elif isinstance(var, ast.arg):
                    var.arg = new_id

        return root, len(selected2synonym)


class TestVarRenamer:
    def test_simple(self):
        code = """
def foo(xxx):
    hello = 134
    foo(xxx)
"""
        root = ast.parse(code)
        root, count = VarRenamer().refactor(root)
        assert count == 1
        assert ast.unparse(root) == "def foo(xxx):\n    welcome = 134\n    foo(xxx)"

    def test_real(self):
        code = """
def load_config(self):
    with open(self.config_path, 'rb') as file:
        dictionary, tf_idf = pickle.load(file)
    return (dictionary, tf_idf)
"""
        root = ast.parse(code)
        root, count = VarRenamer().refactor(root, rand=True)
        assert count == 1

# new code
class CamelCase(RefactorOperator):
    def refactor(
        self, root: ast.AST, rand: bool = False, max_count: int = 1
    ) -> tuple[ast.AST, int]:
        def collect_vars(node: ast.AST, startcollect=False) -> list[ast.Name]:
            if isinstance(node, ast.FunctionDef):
                startcollect = True
            vars = []
            if startcollect:
                if (isinstance(node, ast.Name) and node.id not in ['self', 'super', 'cls']) or \
                   (isinstance(node, ast.arg) and node.arg not in ['self', 'super', 'cls']):
                    vars.append(node)
            decorator_set = frozenset(node.decorator_list) if hasattr(node, 'decorator_list') else frozenset() ##
            for child in ast.iter_child_nodes(node):
                if child not in decorator_set: ##
                    vars += collect_vars(child, startcollect)
            return vars

        vars = collect_vars(root)
        # filter local vars
        name2vars = defaultdict(list)
        for var in vars:
            # don't include _ empty string as candidate
            if (var.id if isinstance(var, ast.Name) else var.arg) == "_":
                continue
            name2vars[(var.id if isinstance(var, ast.Name) else var.arg)].append(var)
        keys = list(name2vars.keys())
        for name in keys:
            vars = [x for x in name2vars[name] if isinstance(x, ast.Name)]
            # local vars are approximated as those that are assigned values
            if not any(map(lambda v: isinstance(v.ctx, ast.Store), vars)):
                del name2vars[name]

        if len(name2vars) == 0:
            return root, 0

        if rand:
            n = min(max_count, len(name2vars))
            names = set(name2vars.keys())
            selected2camel = []

            while 0 <= len(selected2camel) < n and len(names) > 0:
                selected = np.random.choice(list(names), 1, replace=False)[0]
                names.remove(selected)

                #snake_split = selected.split('_')
                #snake_split = list(filter(None, snake_split))
                # if selected is not already Camel case or only contains 1 word
                #if not (re.search(r"^[a-z]+([A-Z][a-z]*)+$", selected) or len(snake_split) <= 1):
                # check if selected is lower snake
                lower_snake = re.search(r"^[a-z]+(_[a-z]+)+$", selected)
                #and len(snake_split) > 1
                if lower_snake is not None :
                    selected2camel.append(selected)

        else:
            keys = list(name2vars.keys())
            selected2camel = []
            for name in keys:
                # check if selected is lower snake
                lower_snake = re.search(r"^[a-z]+(_[a-z]+)+$", name)
                # and len(snake_split) > 1
                if lower_snake is not None:
                    selected2camel.append(name)

        for name in selected2camel:
            vars = name2vars[name]
            # assume var is in lower snake case
            words = name.split('_')
            new_id = words[0]
            # capitalize each word (except the first) that's seperated by _
            for word in words[1:]:
                new_id += word.title()

            print(f"renaming {name} => {new_id}")

            for var in vars:
                if isinstance(var, ast.Name):
                    var.id = new_id
                elif isinstance(var, ast.arg):
                    var.arg = new_id

        return root, len(selected2camel)


class TestCamelCase:
    def test_simple(self):
        code = """
def foo(xxx):
    hello_world = 134
    foo(xxx)
"""
        root = ast.parse(code)
        root, count = CamelCase().refactor(root)
        assert count == 1
        assert ast.unparse(root) == "def foo(xxx):\n    helloWorld = 134\n    foo(xxx)"

    def test_real(self):
        code = """
def load_config(self):
    with open(self.config_path, 'rb') as file:
        dictionary, tf_idf = pickle.load(file)
    return (dictionary, tf_idf)
"""
        root = ast.parse(code)
        root, count = CamelCase().refactor(root, rand=True)
        assert count == 1

    def test_empty(self):
        code = """
def foo(xxx):
    (_, h) = divmod(h, 24)
    foo(xxx)
"""
        root = ast.parse(code)
        root, count = CamelCase().refactor(root)
        assert count == 0
        assert ast.unparse(root) == "def foo(xxx):\n    (_, h) = divmod(h, 24)\n    foo(xxx)"

    def test_trail_empty(self):
        code = """
def foo(xxx):
    lamda_ = np.ones(n_features, dtype=np.float64)
    _lamda = np.ones(n_features, dtype=np.float64)
    foo(xxx)
"""
        root = ast.parse(code)
        root, count = CamelCase().refactor(root)
        assert count == 0
        assert ast.unparse(root) == "def foo(xxx):\n    lamda_ = np.ones(n_features, dtype=np.float64)\n    _lamda = np.ones(n_features, dtype=np.float64)\n    foo(xxx)"


class LowerSnakeCase(RefactorOperator):
    def refactor(
        self, root: ast.AST, rand: bool = False, max_count: int = 1
    ) -> tuple[ast.AST, int]:
        def collect_vars(node: ast.AST, startcollect=False) -> list[ast.Name]:
            if isinstance(node, ast.FunctionDef):
                startcollect = True
            vars = []
            if startcollect:
                if (isinstance(node, ast.Name) and node.id not in ['self', 'super', 'cls']) or \
                   (isinstance(node, ast.arg) and node.arg not in ['self', 'super', 'cls']):
                    vars.append(node)
            decorator_set = frozenset(node.decorator_list) if hasattr(node, 'decorator_list') else frozenset() ##
            for child in ast.iter_child_nodes(node):
                if child not in decorator_set: ##
                    vars += collect_vars(child, startcollect)
            return vars

        vars = collect_vars(root)
        # filter local vars
        name2vars = defaultdict(list)
        for var in vars:
            name2vars[(var.id if isinstance(var, ast.Name) else var.arg)].append(var)
        keys = list(name2vars.keys())
        for name in keys:
            vars = [x for x in name2vars[name] if isinstance(x, ast.Name)]
            # local vars are approximated as those that are assigned values
            if not any(map(lambda v: isinstance(v.ctx, ast.Store), vars)):
                del name2vars[name]

        if len(name2vars) == 0:
            return root, 0

        if rand:
            n = min(max_count, len(name2vars))
            names = set(name2vars.keys())
            selected2snake = []

            while 0 <= len(selected2snake) < n and len(names) > 0:
                selected = np.random.choice(list(names), 1, replace=False)[0]
                names.remove(selected)

                # check if selected is camel
                camel = re.search(r"^[a-z]+([A-Z][a-z]*)+$", selected)
                if camel is not None:
                    selected2snake.append(selected)

        else:
            keys = list(name2vars.keys())
            selected2snake = []
            for name in keys:
                # check if selected is camel
                camel = re.search(r"^[a-z]+([A-Z][a-z]*)+$", name)
                if camel is not None:
                    selected2snake.append(name)

        for name in selected2snake:
            vars = name2vars[name]
            # assume var is in camel case
            words = re.split(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', name)
            new_id = words[0]
            # lower each word
            for word in words[1:]:
                new_id += '_' + word.lower()

            print(f"renaming {name} => {new_id}")

            for var in vars:
                if isinstance(var, ast.Name):
                    var.id = new_id
                elif isinstance(var, ast.arg):
                    var.arg = new_id

        return root, len(selected2snake)


class TestLowerSnakeCase:
    def test_simple(self):
        code = """
def foo(xxx):
    helloWorld = 134
    foo(xxx)
"""
        root = ast.parse(code)
        root, count = LowerSnakeCase().refactor(root)
        assert count == 1
        assert ast.unparse(root) == "def foo(xxx):\n    hello_world = 134\n    foo(xxx)"

    def test_real(self):
        code = """
def load_config(self):
    with open(self.config_path, 'rb') as file:
        dictionary, tfIdf = pickle.load(file)
    return (dictionary, tfIdf)
"""
        root = ast.parse(code)
        root, count = LowerSnakeCase().refactor(root, rand=True)
        assert count == 1

class CommLaw(RefactorOperator):
    def refactor(
        self, root: ast.AST, rand: bool = False, max_count: int = 1
    ) -> tuple[ast.AST, int]:
        assert not rand or rand and max_count > 0
        def collect_and_or(node: ast.AST) -> list[ast.BoolOp]:
            and_ors = []
            if isinstance(node, ast.BoolOp):
                and_ors.append(node)
            for child in ast.iter_child_nodes(node):
                and_ors += collect_and_or(child)
            return and_ors
        #print(ast.dump(root, indent=4))
        and_ors = collect_and_or(root)
        if rand:
            if len(and_ors) != 1:
                selected = (
                    np.random.choice(and_ors, max_count, replace=False) if len(and_ors) > 0 else []
                )
            else: # no random is there's only one candidate
                selected = and_ors
        else:
            selected = and_ors

        #print(selected)
        for node in selected:
            values = node.values

            # shuffle the name variables randomly (can have more than 2 var)
            while True:
                new_values = np.random.permutation(values)
                if np.all(new_values != values):
                    break
            node.values = new_values
        #print(selected)
        return root, len(selected)


class TestCommLaw:
    def test_simple_and(self):
        code = """
if a and b:
    pass
"""
        root = ast.parse(code)
        root, count = CommLaw().refactor(root)
        assert count == 1
        assert ast.unparse(root) == "if b and a:\n    pass"

    def test_simple_or(self):
        code = """
if a or b:
    pass
"""
        root = ast.parse(code)
        root, count = CommLaw().refactor(root)
        assert count == 1
        assert ast.unparse(root) == "if b or a:\n    pass"

    def test_three_and(self):
        code = """
if a and b and c:
    assert True
"""
        root = ast.parse(code)
        root, count = CommLaw().refactor(root)
        assert count == 1
        assert (
            ast.unparse(root)
            == "if b and a and c:\n    assert True"
            or ast.unparse(root)
            == "if b and c and a:\n    assert True"
            or ast.unparse(root)
            == "if c and b and a:\n    assert True"
            or ast.unparse(root)
            == "if c and a and b:\n    assert True"
            or ast.unparse(root)
            == "if a and c and b:\n    assert True"
        )

    def test_three_or(self):
        code = """
if a or b or c:
    assert True
"""
        root = ast.parse(code)
        root, count = CommLaw().refactor(root)
        assert count == 1
        assert (
                ast.unparse(root)
                == "if b or a or c:\n    assert True"
                or ast.unparse(root)
                == "if b or c or a:\n    assert True"
                or ast.unparse(root)
                == "if c or b or a:\n    assert True"
                or ast.unparse(root)
                == "if c or a or b:\n    assert True"
                or ast.unparse(root)
                == "if a or c or b:\n    assert True"
        )

    def test_elif(self):
        code = """
if a and b:
    assert True
elif a or b:
    assert False
else:
    pass
"""
        root = ast.parse(code)
        root, count = CommLaw().refactor(root)
        #print(ast.unparse(root))
        assert count == 2
        assert (
            ast.unparse(root)
            ==
            "if b and a:\n    assert True\nelif b or a:\n    assert False\nelse:\n    pass"
        )

    def test_random(self):
        code = """
if a and b:
    assert True
if a or b:
    assert False
"""
        root = ast.parse(code)
        root, count = CommLaw().refactor(root, rand=True, max_count=1)
        assert count == 1
        assert (
            ast.unparse(root)
            == "if b and a:\n    assert True\nif a or b:\n    assert False"
            or ast.unparse(root)
            == "if a and b:\n    assert True\nif b or a:\n    assert False"
        )

class AddCommLaw(RefactorOperator):
    def __init__(self) -> None:
        super().__init__()
        self.idx = 0

    def refactor(
        self, root: ast.AST, rand: bool = False, max_count: int = 1
    ) -> tuple[ast.AST, int]:
        assert not rand or rand and max_count > 0
        def collect_type_info(node: ast.AST, type_info: dict) -> dict:
            if isinstance(node, ast.Assign):
                # put var name as key & value as content
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        type_info[target.id] = node.value
            # elif isinstance(node, ast.For):
            #     # put var name as key & value as content
            #     for_target = node.target
            #     if isinstance(for_target, ast.Name):
            #         target = for_target.id
            #     elif isinstance(for_target, ast.Tuple):
            #         target = for_target.elts[0]
            #     else:
            #         pass
            #     iter = node.iter
            #     if isinstance(iter, ast.Call):
            #         iter_func = iter.func.id
            #         if iter_func == "range" or iter_func == "enumerate":
            #             type_info[target.id] = ast.Constant()

            for child in ast.iter_child_nodes(node):
                type_info = collect_type_info(child, type_info)
            return type_info

        def collect_add(node: ast.AST, parent: None | ast.AST, type_info: dict) -> list[ast.BinOp]:
            adds = []
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                # if the left and right is Name, check it to exclude list
                if isinstance(node.left, ast.Name):
                    #print(node.left.id)
                    if node.left.id not in type_info:
                        return adds
                    left_type = type_info[node.left.id]
                    #print("left type:", left_type)
                    if not isinstance(left_type, ast.Constant):
                        return adds
                if isinstance(node.right, ast.Name):
                    #print(node.right.id)
                    if node.right.id not in type_info:
                        return adds
                    right_type = type_info[node.right.id]
                    #print("right type:", right_type)
                    if not isinstance(right_type, ast.Constant):
                        return adds

                # only add if + does not belong to another +
                if not isinstance(parent, ast.BinOp):
                    adds.append(node)
                else: # if the left and right is BinOp
                    # only add if + does not belong to another +
                    if not isinstance(parent, ast.BinOp):
                        adds.append(node)

            for child in ast.iter_child_nodes(node):
                adds += collect_add(child, node, type_info)
            return adds
        #print(ast.dump(root, indent=4))
        type_infos = collect_type_info(root, type_info={})
        #print("type_info:", type_infos)
        #print(ast.unparse(root))
        #print(ast.dump(root, indent=4))
        adds = collect_add(root, None, type_infos)
        #print("adds:", adds)
        if rand:
            if len(adds) != 1:
                selected = (
                    np.random.choice(adds, max_count, replace=False) if len(adds) > 0 else []
                )
            else:
                selected = adds
        else:
            selected = adds

        def collect_names(binop: ast.BinOp) -> list[ast.Name]:
            left = binop.left
            right = binop.right
            names = []
            if isinstance(left, ast.BinOp):
                left_name = collect_names(left)
                names.extend(left_name)
            if isinstance(right, ast.BinOp):
                right_name = collect_names(right)
                names.extend(right_name)
            if isinstance(left, ast.Name):
                names.append(left)
            if isinstance(right, ast.Name):
                names.append(right)

            return names

        def write_names(node: ast.AST, names: list[ast.Name]) -> ast.AST:
            if isinstance(node, ast.Name):
                node = names[self.idx]
                self.idx += 1
                return node
            else:
                if isinstance(node, ast.BinOp):
                    node.left = write_names(node.left, names)
                    node.right = write_names(node.right, names)
            return node


        for node in selected:
            names = collect_names(node)
            print("Change order:")
            for name in names:
                print(name.id, end=' ')
            print("=>", end=' ')
            # shuffle the name variables randomly (can have more than 2 var)
            while True:
                new_values = np.random.permutation(names)
                if np.all(new_values != names):
                    break
            for new_name in new_values:
                print(new_name.id, end=' ')
            print()
            #print(ast.dump(node, indent=4))

            node = write_names(node, new_values)
            self.idx = 0
        #print(selected)
        return root, len(selected)


class TestAddCommLaw:
    def test_simple_add(self):
        code = """
def foo(xxx):
    a = 1
    b = 2
    c = a + b
    foo(xxx)
"""
        root = ast.parse(code)
        root, count = AddCommLaw().refactor(root)
        print(count)
        print(ast.unparse(root))
        assert count == 1
        assert ast.unparse(root) == "def foo(xxx):\n    a = 1\n    b = 2\n    c = b + a\n    foo(xxx)"

    def test_two_line_add(self):
        code = """
def foo(xxx):
    a = b = e = d = 1.0
    c = a + b
    f = e + d
    foo(xxx)
"""
        root = ast.parse(code)
        root, count = AddCommLaw().refactor(root)
        print(count)
        print(ast.unparse(root))
        assert count == 2
        assert ast.unparse(root) == "def foo(xxx):\n    a = b = e = d = 1.0\n    c = b + a\n    f = d + e\n    foo(xxx)"

    def test_three_add(self):
        code = """
def foo(xxx):
    a = 1.0
    b = 2.0
    c = 3.0
    d = a + b + c
    foo(xxx)
"""
        root = ast.parse(code)
        root, count = AddCommLaw().refactor(root)
        print(count)
        print(ast.unparse(root))
        assert count == 1
        assert (
            ast.unparse(root)
            == "def foo(xxx):\n    a = 1.0\n    b = 2.0\n    c = 3.0\n    d = b + a + c\n    foo(xxx)"
            or ast.unparse(root)
            == "def foo(xxx):\n    a = 1.0\n    b = 2.0\n    c = 3.0\n    d = b + c + a\n    foo(xxx)"
            or ast.unparse(root)
            == "def foo(xxx):\n    a = 1.0\n    b = 2.0\n    c = 3.0\n    d = c + b + a\n    foo(xxx)"
            or ast.unparse(root)
            == "def foo(xxx):\n    a = 1.0\n    b = 2.0\n    c = 3.0\n    d = c + a + b\n    foo(xxx)"
            or ast.unparse(root)
            == "def foo(xxx):\n    a = 1.0\n    b = 2.0\n    c = 3.0\n    d = a + c + b\n    foo(xxx)"
        )

    def test_four_add(self):
        code = """
def foo(xxx):
    a = b = c = d = 0
    e = a + b + c + d
    foo(xxx)
"""
        root = ast.parse(code)
        root, count = AddCommLaw().refactor(root)
        print(count)
        print(ast.unparse(root))
        assert count == 1

    def test_list(self):
        code = """
def foo(xxx):
    arr_1 = [1, 2]
    arr_2 = [3, 4]
    int_1 = 2
    int_2 = 3
    float_1 = 1.1
    float_2 = 2.2
    arr_3 = arr_1 + arr_2
    int_3 = int_1 + int_2
    float_3 = float_1 + float_2
"""
        root = ast.parse(code)
        root, count = AddCommLaw().refactor(root)
        print(count)
        print(ast.unparse(root))
        assert count == 2
        # don't change list position
        assert (
            ast.unparse(root)
            == "def foo(xxx):\n    arr_1 = [1, 2]\n    arr_2 = [3, 4]\n    int_1 = 2\n    int_2 = 3\n    float_1 = 1.1\n    float_2 = 2.2\n    arr_3 = arr_1 + arr_2\n    int_3 = int_2 + int_1\n    float_3 = float_2 + float_1"
        )

    def test_random(self):
        code = """
def foo(xxx):
    a = b = 1
    e = d = 1.0
    c = a + b
    f = e + d
    foo(xxx)
"""
        root = ast.parse(code)
        root, count = AddCommLaw().refactor(root, rand=True, max_count=1)
        print(count)
        print(ast.unparse(root))
        assert count == 1
        assert (
            ast.unparse(root)
            == "def foo(xxx):\n    a = b = 1\n    e = d = 1.0\n    c = b + a\n    f = e + d\n    foo(xxx)"
            or ast.unparse(root)
            == "def foo(xxx):\n    a = b = 1\n    e = d = 1.0\n    c = a + b\n    f = d + e\n    foo(xxx)"
        )

class List2Range(RefactorOperator):
    def refactor(
        self, root: ast.AST, rand: bool = False, max_count: int = 1
    ) -> tuple[ast.AST, int]:
        assert not rand or rand and max_count > 0
        def collect_for(node: ast.AST) -> list[ast.For]:
            ands = []
            if isinstance(node, ast.For):
                # don't collect for-loop with call function or list or constant as array, or target with tuple
                #if isinstance(node.iter, ast.Call) or isinstance(node.iter, ast.List) or isinstance(node.iter, ast.Constant) or isinstance(node.target, ast.Tuple):
                # only collect for-loop with name or attribute as array, & don't collect target with tuple
                if not (isinstance(node.iter, ast.Name) or isinstance(node.iter, ast.Attribute)) or isinstance(node.target, ast.Tuple):
                    #if isinstance(node.iter.func, ast.Name):
                        #if node.iter.func.id == "range":
                    return ands
                ands.append(node)
            for child in ast.iter_child_nodes(node):
                ands += collect_for(child)
            return ands

        fors = collect_for(root)
        if rand:
            if len(fors) != 1:
                selected = (
                    np.random.choice(fors, max_count, replace=False) if len(fors) > 0 else []
                )
            else: # no random is there's only one candidate
                selected = fors
        else:
            selected = fors

        def collect_vars(node: ast.AST, parent: ast.AST | None, field: str | None, it: ast.Name | None) -> list[(ast.Name, ast.Name, str)]:
            vars = []
            # can be used to collect list iter or local var for any node
            if it is not None:
                if node == it: # don't collect iter var obj
                    return vars

            if isinstance(node, ast.Name):
                # only collect local var with same id as iter
                if it is not None:
                    if node.id == it.id:
                        vars.append((node, parent, field))
                else:
                    vars.append((node, parent, field))


            for field, child_list in ast.iter_fields(node):
                #print(f"field: {field}, child: {child_list}")
                if not isinstance(child_list, list):
                    child_list = [child_list]
                #print("list_child:", child_list)
                for child in child_list:
                    if isinstance(child, ast.AST):
                        # for child in list_child:
                        vars += collect_vars(child, node, field, it)
            return vars

        def collect_tree(node: ast.AST, parent: ast.AST | None, field: str | None, vars: dict) -> dict:
            # collect all nodes with parent infos
            vars[node] = (parent, field)

            for field, child_list in ast.iter_fields(node):
                #print(f"field: {field}, child: {child_list}")
                if not isinstance(child_list, list):
                    child_list = [child_list]
                #print("list_child:", child_list)s
                for child in child_list:
                    if isinstance(child, ast.AST):
                        # for child in list_child:
                        collect_tree(child, node, field, vars)
            return vars

        tree = collect_tree(root, None, None, vars={})
        for node in selected:
            #print(ast.unparse(node))
            # get info
            iter_arr = node.iter # array
            target = node.target # item
            arr = {}

            # get arr
            arr[target.id] = collect_vars(iter_arr, node, "target", None)
            #print(f"{target.id}: {arr[target.id]}")
            #print("arr:", arr)
            #list_name = node.iter

            # get local var with parent and field info
            vars_list = []
            for stmt in node.body:
                vars_list.extend(collect_vars(stmt, None, None, target))

            #for var, parent, field in vars_list:
            #    print("var:", ast.unparse(var), "parent:", parent, "field:", field)

            # modify for statement
            range_func= ast.Name(id='range', ctx=ast.Load())
            len_func = ast.Name(id='len', ctx=ast.Load())

            range_args = [ast.Call(func=len_func, args=[node.iter], keywords=[])]
            node.iter = ast.Call(func=range_func, args=range_args, keywords=[])

            # use the original variable as idx, ensure uniqueness
            #idx = ast.Name(id=f'{list_iter.id}', ctx=ast.Load())
            #node.target = idxs
            #print("arr", arr)

            #field, child_list = ast.iter_fields(node)
            # modify local var
            for var, parent, field in vars_list:
                # idx = ast.Name(id='i', ctx=ast.Load())
                #print(f"arr[{var.id}]: {{{arr[var.id]}}}")
                arr_name, arr_parent, arr_field = arr[var.id][0]
                # if arr_name parent isn't For node = it's inside attribute
                while not isinstance(arr_parent, ast.For):
                    arr_name = arr_parent
                    #print("arr_parent:", arr_parent, ast.unparse(arr_parent))
                    arr_parent = tree[arr_parent][0]
                #setattr(arr_parent, arr_field, var)
                #arr_name = ast.Name(id=f'{arr[var.id]}', ctx=ast.Load())
                # use the original variable as idx, ensure uniqueness
                idx = ast.Name(id=var.id, ctx=ast.Load())
                new_var = ast.Subscript(value=arr_name, slice=idx, ctx=var.ctx)

                # update tree
                del tree[var]
                tree[new_var] = (parent, field)

                # if the attribute is a list, make into lists
                if isinstance(getattr(parent, field), list):
                    new_var = [new_var]
                setattr(parent, field, new_var)

        #print(ast.dump(root, indent=2))
        return root, len(selected)


class TestList2Range:
    def test_simple(self):
        code = """
arr = [1, 2]
for item in arr:
    pass
"""
        root = ast.parse(code)
        root, count = List2Range().refactor(root)
        print(ast.unparse(root))
        assert count == 1
        assert (
            ast.unparse(root)
            == "arr = [1, 2]\nfor item in range(len(arr)):\n    pass"
        )

    def test_local_var(self):
        code = """
arr = [1, 2]
for item in arr:
    item = 0
"""
        root = ast.parse(code)
        root, count = List2Range().refactor(root)
        print(ast.unparse(root))
        assert count == 1
        assert (
            ast.unparse(root)
            == "arr = [1, 2]\nfor item in range(len(arr)):\n    arr[item] = 0"
        )

    def test_nested_for(self):
        code = """
arr_1 = [1, 2]
arr_2 = [3, 4]
for item_1 in arr_1:
    for i in arr_2:
        item_1 = i + 1
        i = item_1 + 1
"""
        root = ast.parse(code)
        root, count = List2Range().refactor(root)
        print(ast.unparse(root))
        assert count == 2
        assert (
                ast.unparse(root)
                == "arr_1 = [1, 2]\narr_2 = [3, 4]\nfor item_1 in range(len(arr_1)):\n    for i in range(len(arr_2)):\n        arr_1[item_1] = arr_2[i] + 1\n        arr_2[i] = arr_1[item_1] + 1"
        )

    def test_random(self):
        code = """
arr_1 = [1, 2]
arr_2 = [3, 4]
for item in arr_1:
    for i in arr_2:
        item = 0
        i = 1
"""
        root = ast.parse(code)
        root, count = List2Range().refactor(root, rand=True, max_count=1)
        print(ast.unparse(root))
        assert count == 1
        assert (
                ast.unparse(root)
                == "arr_1 = [1, 2]\narr_2 = [3, 4]\nfor item in arr_1:\n    for i in range(len(arr_2)):\n        item = 0\n        arr_2[i] = 1"
                or ast.unparse(root)
                == "arr_1 = [1, 2]\narr_2 = [3, 4]\nfor item in range(len(arr_1)):\n    for i in arr_2:\n        arr_1[item] = 0\n        i = 1"
        )

    def test_range(self):
        code = """
arr = [1, 2]
for item in range(len(arr)):
    arr[item] = 0
"""
        root = ast.parse(code)
        root, count = List2Range().refactor(root)
        print(ast.unparse(root))
        assert count == 0
        assert (
            ast.unparse(root)
            == "arr = [1, 2]\nfor item in range(len(arr)):\n    arr[item] = 0"
        )

    def test_func(self):
        code = """
arr = (1, 2)
arr_list = list(arr)
for item in arr_list:
    item = 0
"""
        root = ast.parse(code)
        root, count = List2Range().refactor(root)
        print(ast.unparse(root))
        assert count == 1
        assert (
                ast.unparse(root)
                == "arr = (1, 2)\narr_list = list(arr)\nfor item in range(len(arr_list)):\n    arr_list[item] = 0"
        )

    def test_attr(self):
        code = """
for mapping in self.maps:
    if key in mapping:
        mapping[key] = value
"""
        root = ast.parse(code)
        root, count = List2Range().refactor(root)
        print(ast.unparse(root))
        assert count == 1
        assert (
                ast.unparse(root)
                == "for mapping in range(len(self.maps)):\n    if key in self.maps[mapping]:\n        self.maps[mapping][key] = value"
        )

    def test_attr2(self):
        code = """
for mapping in self.maps.id:
    if key in mapping.id:
        mapping[key] = value
"""
        root = ast.parse(code)
        root, count = List2Range().refactor(root)
        print(ast.unparse(root))
        assert count == 1
        assert (
                ast.unparse(root)
                == "for mapping in range(len(self.maps.id)):\n    if key in self.maps.id[mapping].id:\n        self.maps.id[mapping][key] = value"
        )

    def test_tuple(self):
        code = """
list = [1, 2]
tuple_list = list(zip(list, range(3)))
for item, x in list:
    print(item)
    print(x)
for item, x in zip(list, range(3)):
    print(item)
    print(x)
for i, item in enumerate(list):
    print(x)
    print(item)
"""
        root = ast.parse(code)
        root, count = List2Range().refactor(root)
        print(ast.unparse(root))
        assert count == 0
        assert (
                ast.unparse(root)
                == "list = [1, 2]\ntuple_list = list(zip(list, range(3)))\nfor (item, x) in list:\n    print(item)\n    print(x)\nfor (item, x) in zip(list, range(3)):\n    print(item)\n    print(x)\nfor (i, item) in enumerate(list):\n    print(x)\n    print(item)"
        )


class CamelSnakeExchange(RefactorOperator):
    def refactor(
        self, root: ast.AST, rand: bool = False, max_count: int = 1
    ) -> tuple[ast.AST, int]:
        def collect_vars(node: ast.AST, startcollect=False) -> list[ast.Name]:
            if isinstance(node, ast.FunctionDef):
                startcollect = True
            vars = []
            if startcollect:
                if (isinstance(node, ast.Name) and node.id not in ['self', 'super', 'cls']) or \
                   (isinstance(node, ast.arg) and node.arg not in ['self', 'super', 'cls']):
                    vars.append(node)
            decorator_set = frozenset(node.decorator_list) if hasattr(node, 'decorator_list') else frozenset() ##
            for child in ast.iter_child_nodes(node):
                if child not in decorator_set: ##
                    vars += collect_vars(child, startcollect)
            return vars

        vars = collect_vars(root)
        # filter local vars
        name2vars = defaultdict(list)
        for var in vars:
            # don't include _ empty string as candidate
            if (var.id if isinstance(var, ast.Name) else var.arg) == "_":
                continue
            name2vars[(var.id if isinstance(var, ast.Name) else var.arg)].append(var)
        keys = list(name2vars.keys())
        for name in keys:
            vars = [x for x in name2vars[name] if isinstance(x, ast.Name)]
            # local vars are approximated as those that are assigned values
            if not any(map(lambda v: isinstance(v.ctx, ast.Store), vars)):
                del name2vars[name]

        if len(name2vars) == 0:
            return root, 0

        if rand:
            n = min(max_count, len(name2vars))
            names = set(name2vars.keys())
            selected2camel = []
            selected2snake = []

            while 0 <= len(selected2camel) < n and len(names) > 0:
                selected = np.random.choice(list(names), 1, replace=False)[0]
                names.remove(selected)

                # check if selected is lower snake
                lower_snake = re.search(r"^[a-z]+(_[a-z]+)+$", selected)
                # and len(snake_split) > 1
                if lower_snake is not None:
                    selected2camel.append(selected)

                else: # camel and snake should be exclusive
                    # check if selected is camel
                    camel = re.search(r"^[a-z]+([A-Z][a-z]*)+$", selected)
                    if camel is not None:
                        selected2snake.append(selected)

        else:
            keys = list(name2vars.keys())
            selected2camel = []
            for name in keys:
                # check if selected is lower snake
                lower_snake = re.search(r"^[a-z]+(_[a-z]+)+$", name)
                # and len(snake_split) > 1
                if lower_snake is not None:
                    selected2camel.append(name)
            selected2snake = []
            for name in keys:
                # check if selected is camel
                camel = re.search(r"^[a-z]+([A-Z][a-z]*)+$", name)
                if camel is not None:
                    selected2snake.append(name)

        # print("to camel:", selected2camel)
        # print("to snake:", selected2snake)
        # rename lower snake to camel
        for name in selected2camel:
            vars = name2vars[name]
            # assume var is in lower snake case
            words = name.split('_')
            new_id = words[0]
            # capitalize each word (except the first) that's seperated by _
            for word in words[1:]:
                new_id += word.title()

            print(f"renaming {name} => {new_id}")

            for var in vars:
                if isinstance(var, ast.Name):
                    var.id = new_id
                elif isinstance(var, ast.arg):
                    var.arg = new_id

        # rename camel to lower snake
        for name in selected2snake:
            vars = name2vars[name]
            # assume var is in camel case
            words = re.split(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', name)
            new_id = words[0]
            # lower each word
            for word in words[1:]:
                new_id += '_' + word.lower()

            print(f"renaming {name} => {new_id}")

            for var in vars:
                if isinstance(var, ast.Name):
                    var.id = new_id
                elif isinstance(var, ast.arg):
                    var.arg = new_id

        selected_camel_snake = len(selected2camel) + len(selected2snake)

        return root, selected_camel_snake

class TestCamelSnakeExchange:

    def test_simple_snake(self):
        code = """
def foo(xxx):
    helloWorld = 134
    foo(xxx)
"""
        root = ast.parse(code)
        root, count = CamelSnakeExchange().refactor(root)
        assert count == 1
        assert ast.unparse(root) == "def foo(xxx):\n    hello_world = 134\n    foo(xxx)"

    def test_real_snake(self):
        code = """
def load_config(self):
    with open(self.config_path, 'rb') as file:
        dictionary, tfIdf = pickle.load(file)
    return (dictionary, tfIdf)
"""
        root = ast.parse(code)
        root, count = CamelSnakeExchange().refactor(root, rand=True)
        assert count == 1

    def test_simple_camel(self):
        code = """
def foo(xxx):
    hello_world = 134
    foo(xxx)
"""
        root = ast.parse(code)
        root, count = CamelSnakeExchange().refactor(root)
        assert count == 1
        assert ast.unparse(root) == "def foo(xxx):\n    helloWorld = 134\n    foo(xxx)"

    def test_real_camel(self):
        code = """
def load_config(self):
    with open(self.config_path, 'rb') as file:
        dictionary, tf_idf = pickle.load(file)
    return (dictionary, tf_idf)
"""
        root = ast.parse(code)
        root, count = CamelSnakeExchange().refactor(root, rand=True)
        assert count == 1

    def test_empty_camel(self):
        code = """
def foo(xxx):
    (_, h) = divmod(h, 24)
    foo(xxx)
"""
        root = ast.parse(code)
        root, count = CamelSnakeExchange().refactor(root)
        assert count == 0
        assert ast.unparse(root) == "def foo(xxx):\n    (_, h) = divmod(h, 24)\n    foo(xxx)"

    def test_trail_empty_camel(self):
        code = """
def foo(xxx):
    lamda_ = np.ones(n_features, dtype=np.float64)
    _lamda = np.ones(n_features, dtype=np.float64)
    foo(xxx)
"""
        root = ast.parse(code)
        root, count = CamelSnakeExchange().refactor(root)
        assert count == 0
        assert ast.unparse(
            root) == "def foo(xxx):\n    lamda_ = np.ones(n_features, dtype=np.float64)\n    _lamda = np.ones(n_features, dtype=np.float64)\n    foo(xxx)"

    def test_snake_camel(self):
        code = """
def foo(xxx):
    lamda = 1
    zLamda = 0
    lamda_i = lamda + zLamda
    foo(xxx)
"""
        root = ast.parse(code)
        root, count = CamelSnakeExchange().refactor(root)
        #print(count)
        #print(ast.unparse(root))
        assert count == 2
        assert ast.unparse(
            root) == "def foo(xxx):\n    lamda = 1\n    z_lamda = 0\n    lamdaI = lamda + z_lamda\n    foo(xxx)"

    def test_args(self):
        code = """
def foo(lamda, zLamda, lamda_i):
    lamda = 1
    zLamda = 0
    lamda_i = lamda + zLamda
    foo(lamda, zLamda, lamda_i)
"""
        root = ast.parse(code)
        root, count = CamelSnakeExchange().refactor(root)
        # print(count)
        # print(ast.unparse(root))
        assert count == 2
        assert ast.unparse(
            root) == "def foo(lamda, z_lamda, lamdaI):\n    lamda = 1\n    z_lamda = 0\n    lamdaI = lamda + z_lamda\n    foo(lamda, z_lamda, lamdaI)"
