from abc import ABC
from typing import Generic, TypeVar
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch
import ast
from nltk.translate.bleu_score import sentence_bleu
import edist.sed as sed
import edist.ted as ted
import networkx as nx

T = TypeVar("T")


class Similaity(ABC, Generic[T]):
    def calculate(self, one: T, other: T) -> float:
        pass


class StringEditDistance(Similaity[str]):
    def calculate(self, one: str, other: str) -> float:
        distance = sed.standard_sed(one, other)
        sim = 1 - distance / max(len(one), len(other))
        return sim


class TestStringEditDistance:
    def test_same(self):
        one = "def add(a, b):\n    return a + b"
        other = "def add(a, b):\n    return a + b"
        sim = StringEditDistance().calculate(one, other)
        assert sim == 1

    def test_dissimilar(self):
        one = "def add(a, b):\n    return a + b"
        other = "def print_hello_world():\n    print('Hello, World!')"
        sim = StringEditDistance().calculate(one, other)
        assert sim < 1


class CosineSimilarity(Similaity[np.ndarray]):
    def calculate(self, one: np.ndarray, other: np.ndarray) -> float:
        assert len(one.shape) == 1, "one dimension only for cosine similarity"
        assert one.shape == other.shape
        return np.dot(one, other) / (np.linalg.norm(one) * np.linalg.norm(other))


class TestCosineSimilarity:
    def test_similar(self):
        one = np.array([1, 2, 3])
        other = np.array([1, 2, 3])
        sim = CosineSimilarity().calculate(one, other)
        assert sim == 1

    def test_dissimilar(self):
        one = np.array([1, 2, 3])
        other = np.array([1, 2, 4])
        sim = CosineSimilarity().calculate(one, other)
        assert sim < 1


class CodeBert(Similaity[str]):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self._cosine_similarity = CosineSimilarity()

    def code2vec(self, code: str) -> np.ndarray:
        inputs = self.tokenizer(
            code, return_tensors="pt", max_length=512, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[0][0]
        code_vec = embedding.detach().numpy()
        return code_vec

    def calculate(self, one: str, other: str) -> float:
        return self._cosine_similarity.calculate(
            self.code2vec(one),
            self.code2vec(other),
        )


class TestCodeBert:
    def test_same(self):
        one_code = "def add(a, b):\n    return a + b"
        other_code = "def add(a, b):\n    return a + b"
        sim = CodeBert().calculate(one_code, other_code)
        # add a small tolerance
        assert sim > 0.999

    def test_dissimilar(self):
        one_code = "def add(a, b):\n    return a + b"
        other_code = "def print_hello_world():\n    print('Hello, World!')"
        sim = CodeBert().calculate(one_code, other_code)
        assert sim < 1


class GraphCodeBert(Similaity[str]):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        self.model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
        self._cosine_similarity = CosineSimilarity()

    def code2vec(self, code: str) -> np.ndarray:
        inputs = self.tokenizer(
            code, return_tensors="pt", max_length=512, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[0][0]
        code_vec = embedding.detach().numpy()
        return code_vec

    def calculate(self, one: str, other: str) -> float:
        return self._cosine_similarity.calculate(
            self.code2vec(one),
            self.code2vec(other),
        )


class TestGraphCodeBert:
    def test_dissimilar(self):
        one_code = "def add(a, b):\n    return a + b"
        other_code = "def print_hello_world():\n    print('Hello, World!')"
        sim = GraphCodeBert().calculate(one_code, other_code)
        assert sim < 0.9


class CodeT5(Similaity[str]):
    def __init__(self) -> None:
        super().__init__()
        checkpoint = "Salesforce/codet5p-110m-embedding"
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(
            "cuda"
        )
        self._cosine_similarity = CosineSimilarity()

    def code2vec(self, code: str) -> np.ndarray:
        inputs = self.tokenizer.encode(
            code, return_tensors="pt", max_length=512, truncation=True
        ).to("cuda")
        embedding = self.model(inputs)[0].cpu()
        code_vec = embedding.detach().numpy()
        return code_vec

    def calculate(self, one: str, other: str) -> float:
        return self._cosine_similarity.calculate(
            self.code2vec(one),
            self.code2vec(other),
        )


class TestCodeT5:
    def test_dissimilar(self):
        one_code = "def add(a, b):\n    return a + b"
        other_code = "def print_hello_world():\n    print('Hello, World!')"
        sim = CodeT5().calculate(one_code, other_code)
        assert sim < 0.9

    def test_similar(self):
        one_code = "def add(a, b):\n    return a + b"
        other_code = "def add(a, b):\n    return a + b"
        sim = CodeT5().calculate(one_code, other_code)
        assert sim > 0.99


class BLEU(Similaity[str]):
    def calculate(self, one: str, other: str) -> float:
        # Tokenize the reference and candidate code
        reference_tokens = one.split()
        candidate_tokens = other.split()

        # Compute BLEU score
        weights = tuple(1.0 / 4 for _ in range(4))
        bleu_score = sentence_bleu(
            [reference_tokens], candidate_tokens, weights=weights
        )

        return bleu_score


class TesteBLEU:
    def test_dissimilar(self):
        one_code = "def add(a, b):\n    return a + b"
        other_code = "def print_hello_world():\n    print('Hello, World!')"
        sim = BLEU().calculate(one_code, other_code)
        assert sim < 1e-10

    def test_similar(self):
        one_code = "def add(a, b):\n    return a + b"
        other_code = "def add(a, b):\n    return a - b"
        sim = BLEU().calculate(one_code, other_code)
        assert sim > 0.6


class ASTEditDistance(Similaity[ast.AST]):
    """
    The tree edit distance of AST using Zhang-Shasha algorithm
    """

    def ast2adj(self, root: ast.AST) -> tuple[list[str], list[list[int]]]:
        def node2label(node: ast.AST) -> str:
            if len([None for _ in ast.iter_child_nodes(node)]) == 0:
                return ast.unparse(node)
            else:
                return type(node).__name__

        graph = nx.DiGraph()
        for node in ast.walk(root):
            graph.add_node(id(node), label=node2label(node))
            # depth-first traversal
            for child in ast.iter_child_nodes(node):
                graph.add_edge(id(node), id(child))
        nodes = [graph.nodes[node]["label"] for node in graph.nodes]
        node2idx = {node: idx for idx, node in enumerate(graph.nodes)}
        adj_matrix = []
        for node in graph.nodes:
            adj = []
            for child in graph.successors(node):
                adj.append(node2idx[child])
            adj_matrix.append(adj)
        return (nodes, adj_matrix)

    def calculate(self, one: ast.AST, other: ast.AST) -> float:
        one_nodes, one_adj = self.ast2adj(one)
        other_nodes, other_adj = self.ast2adj(other)
        distance = ted.standard_ted(one_nodes, one_adj, other_nodes, other_adj)
        sim = 1 - distance / max(len(one_nodes), len(other_nodes))
        return sim


class TestASTEditDistance:
    def test_ast2adj(self):
        code = """
import pathlib
print(123)
"""
        (nodes, adj) = ASTEditDistance().ast2adj(ast.parse(code))
        assert nodes == [
            "Module",
            "Import",
            "Expr",
            "pathlib",
            "Call",
            "Name",
            "123",
            "",
        ]
        assert adj == [[1, 2], [3], [4], [], [5, 6], [7], [], []]

    def test_dissimilar(self):
        one_code = """
print(123)
"""
        other_code = """
import abc
"""
        one_ast = ast.parse(one_code)
        other_ast = ast.parse(other_code)
        sim = ASTEditDistance().calculate(one_ast, other_ast)
        assert sim < 1

    def test_same(self):
        one_code = "def add(a, b):\n    return a + b"
        other_code = "def add(a, b):\n    return a + b"
        one_ast = ast.parse(one_code)
        other_ast = ast.parse(other_code)
        sim = ASTEditDistance().calculate(one_ast, other_ast)
        assert sim > 0.999


class JaccardSimilarity(Similaity):
    """
    Jaccard sim approximated with minHashes.
    We use unigram as shingling.
    """

    def calculate(self, one: ast.AST, other: ast.AST) -> float:
        def ast2shinglings(root: ast.AST) -> set[str]:
            text = ast.unparse(root)
            words = text.split()
            words = map(lambda x: x.strip(), words)
            words = filter(lambda x: len(x) > 0, words)
            words = set(words)
            return words

        from datasketch import MinHash

        one_shinglings = ast2shinglings(one)
        other_shinglings = ast2shinglings(other)
        m1, m2 = MinHash(), MinHash()
        for d in one_shinglings:
            m1.update(d.encode("utf8"))
        for d in other_shinglings:
            m2.update(d.encode("utf8"))
        return m1.jaccard(m2)


class TestJaccardSimilarity:
    def test_dissimilar(self):
        one_code = """
def add(a, b):
    return a + b
"""
        other_code = """
def mul(a, b):
    return a * b
"""
        sim = JaccardSimilarity().calculate(ast.parse(one_code), ast.parse(other_code))
        assert sim < 0.99

    def test_dissimilar(self):
        one_code = """
def add(a, b):
    return a + b
"""
        sim = JaccardSimilarity().calculate(ast.parse(one_code), ast.parse(one_code))
        assert sim == 1