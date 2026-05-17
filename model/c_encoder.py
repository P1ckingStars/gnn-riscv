"""C source → graph → GNN-encoded function representation.

Pipeline:
  1. Parse a single C function with libclang into its AST cursor tree.
  2. Walk the AST, flattening every cursor into a graph node with per-node features
     (cursor kind, type class, literal-value bucket, depth).
  3. Add typed edges:
       - ``AST_CHILD`` — parent → child
       - ``DATA_REF``  — DECL_REF_EXPR → the VAR_DECL/PARM_DECL it references
       - ``NEXT_STMT`` — sequential order within a CompoundStmt (control flow)
  4. PyG ``GATConv`` stack refines node embeddings; attention-pool produces a
     function-level context vector + per-node embeddings used as decoder cross-
     attention keys/values.

The extraction layer (``parse_c_function``) is pure-Python and can be tested without
PyTorch. The ``CEncoder`` ``nn.Module`` is the only PyG-dependent piece.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

from clang import cindex


# ---- Vocabularies -----------------------------------------------------------

# Subset of clang CursorKinds we expect to see in LeetCode-style C. Anything else
# goes to UNKNOWN_KIND.
_INTERESTING_KINDS = [
    "FUNCTION_DECL", "PARM_DECL", "VAR_DECL", "FIELD_DECL", "STRUCT_DECL",
    "TYPEDEF_DECL",
    "COMPOUND_STMT", "DECL_STMT", "RETURN_STMT",
    "IF_STMT", "WHILE_STMT", "FOR_STMT", "DO_STMT",
    "SWITCH_STMT", "CASE_STMT", "DEFAULT_STMT", "BREAK_STMT", "CONTINUE_STMT",
    "GOTO_STMT", "LABEL_STMT", "NULL_STMT",
    "BINARY_OPERATOR", "UNARY_OPERATOR", "COMPOUND_ASSIGNMENT_OPERATOR",
    "CONDITIONAL_OPERATOR", "CSTYLE_CAST_EXPR", "PAREN_EXPR",
    "INTEGER_LITERAL", "FLOATING_LITERAL", "CHARACTER_LITERAL", "STRING_LITERAL",
    "DECL_REF_EXPR", "MEMBER_REF_EXPR", "ARRAY_SUBSCRIPT_EXPR", "CALL_EXPR",
    "INIT_LIST_EXPR", "UNEXPOSED_EXPR",
]


class CursorKindVocab:
    UNKNOWN_KIND = 0  # also doubles as PAD

    def __init__(self) -> None:
        self._name_to_idx = {n: i + 1 for i, n in enumerate(_INTERESTING_KINDS)}
        self.size = len(self._name_to_idx) + 1  # +1 for UNKNOWN

    def encode(self, kind_name: str) -> int:
        return self._name_to_idx.get(kind_name, self.UNKNOWN_KIND)


_TYPE_CLASSES = ["VOID", "INT", "FLOAT", "POINTER", "STRUCT", "ARRAY", "UNKNOWN"]


class TypeClassVocab:
    def __init__(self) -> None:
        self._name_to_idx = {n: i for i, n in enumerate(_TYPE_CLASSES)}
        self.size = len(_TYPE_CLASSES)
        self.UNKNOWN = self._name_to_idx["UNKNOWN"]

    def encode(self, ty: cindex.Type) -> int:
        k = ty.kind
        # Integer kinds
        int_kinds = {
            cindex.TypeKind.BOOL, cindex.TypeKind.CHAR_U, cindex.TypeKind.UCHAR,
            cindex.TypeKind.CHAR_S, cindex.TypeKind.SCHAR,
            cindex.TypeKind.USHORT, cindex.TypeKind.UINT, cindex.TypeKind.ULONG,
            cindex.TypeKind.ULONGLONG,
            cindex.TypeKind.SHORT, cindex.TypeKind.INT, cindex.TypeKind.LONG,
            cindex.TypeKind.LONGLONG,
        }
        if k in int_kinds:
            return self._name_to_idx["INT"]
        if k in (cindex.TypeKind.FLOAT, cindex.TypeKind.DOUBLE, cindex.TypeKind.LONGDOUBLE):
            return self._name_to_idx["FLOAT"]
        if k == cindex.TypeKind.VOID:
            return self._name_to_idx["VOID"]
        if k in (cindex.TypeKind.POINTER,):
            return self._name_to_idx["POINTER"]
        if k in (cindex.TypeKind.RECORD,):
            return self._name_to_idx["STRUCT"]
        if k in (cindex.TypeKind.CONSTANTARRAY, cindex.TypeKind.INCOMPLETEARRAY,
                 cindex.TypeKind.VARIABLEARRAY):
            return self._name_to_idx["ARRAY"]
        return self.UNKNOWN


# Per-node literal-value buckets (covers most literals we see). Reuses the same
# bucketing logic as ConstVocab in spirit.
_VALUE_BUCKETS = [
    -1024, -512, -256, -128, -64, -32, -16, -8,
    -4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 16, 32, 64, 100, 128, 256, 1000, 1024, 4096, 10000,
    0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
]


class ValueBucketVocab:
    OOV = 0

    def __init__(self) -> None:
        self._val_to_idx = {v: i + 1 for i, v in enumerate(_VALUE_BUCKETS)}
        self.size = len(_VALUE_BUCKETS) + 1

    def encode(self, value: int) -> int:
        return self._val_to_idx.get(int(value), self.OOV)


# ---- Graph extraction -------------------------------------------------------

class EdgeType(Enum):
    AST_CHILD = 0
    DATA_REF = 1
    NEXT_STMT = 2


@dataclass
class CFnNode:
    idx: int
    kind_name: str
    type_class: int
    has_value: bool
    value_bucket: int
    depth: int
    spelling: str
    cursor: object = field(repr=False)  # cindex.Cursor


@dataclass
class CFnEdge:
    src: int
    dst: int
    type: EdgeType


@dataclass
class CFunctionGraph:
    fn_name: str
    nodes: list[CFnNode]
    edges: list[CFnEdge]

    def n_nodes(self) -> int:
        return len(self.nodes)

    def n_edges(self) -> int:
        return len(self.edges)

    def edges_of_type(self, t: EdgeType) -> Iterable[CFnEdge]:
        return (e for e in self.edges if e.type is t)


_index: Optional[cindex.Index] = None


def _idx() -> cindex.Index:
    global _index
    if _index is None:
        _index = cindex.Index.create()
    return _index


def parse_c_function(
    c_source: str,
    fn_name: str,
    kind_vocab: CursorKindVocab,
    type_vocab: TypeClassVocab,
    value_vocab: ValueBucketVocab,
) -> CFunctionGraph:
    """Parse ``c_source`` and extract the graph for the function named ``fn_name``."""
    tu = _idx().parse(
        "x.c", args=["-xc", "-std=c11"], unsaved_files=[("x.c", c_source)],
    )
    if tu is None:
        raise ValueError("clang parse failed")
    fn_cursor = None
    for c in tu.cursor.walk_preorder():
        if (
            c.kind == cindex.CursorKind.FUNCTION_DECL
            and c.is_definition()
            and c.spelling == fn_name
        ):
            fn_cursor = c
            break
    if fn_cursor is None:
        raise ValueError(f"function {fn_name!r} not found in source")

    nodes: list[CFnNode] = []
    edges: list[CFnEdge] = []
    cursor_to_idx: dict[int, int] = {}   # cursor.hash → node index

    def add_node(cur: cindex.Cursor, depth: int) -> int:
        h = cur.hash
        if h in cursor_to_idx:
            return cursor_to_idx[h]
        idx = len(nodes)
        has_value = cur.kind == cindex.CursorKind.INTEGER_LITERAL
        value_bucket = 0
        if has_value:
            try:
                tok = next(cur.get_tokens(), None)
                if tok is not None:
                    raw = tok.spelling.rstrip("ulUL")
                    v = int(raw, 0)
                    value_bucket = value_vocab.encode(v)
            except (StopIteration, ValueError):
                value_bucket = 0
        node = CFnNode(
            idx=idx,
            kind_name=cur.kind.name,
            type_class=type_vocab.encode(cur.type),
            has_value=has_value,
            value_bucket=value_bucket,
            depth=depth,
            spelling=(cur.spelling or "")[:32],
            cursor=cur,
        )
        nodes.append(node)
        cursor_to_idx[h] = idx
        return idx

    def visit(cur: cindex.Cursor, parent_idx: Optional[int], depth: int) -> int:
        i = add_node(cur, depth)
        if parent_idx is not None:
            edges.append(CFnEdge(parent_idx, i, EdgeType.AST_CHILD))
        # DATA_REF: DECL_REF_EXPR → referenced decl.
        if cur.kind == cindex.CursorKind.DECL_REF_EXPR:
            ref = cur.referenced
            if ref is not None and ref.hash in cursor_to_idx:
                edges.append(CFnEdge(i, cursor_to_idx[ref.hash], EdgeType.DATA_REF))
        # Sequential edges for direct children of a CompoundStmt.
        child_indices: list[int] = []
        for ch in cur.get_children():
            child_indices.append(visit(ch, i, depth + 1))
        if cur.kind == cindex.CursorKind.COMPOUND_STMT:
            for a, b in zip(child_indices, child_indices[1:]):
                edges.append(CFnEdge(a, b, EdgeType.NEXT_STMT))
        return i

    visit(fn_cursor, parent_idx=None, depth=0)
    return CFunctionGraph(fn_name=fn_name, nodes=nodes, edges=edges)


# ---- PyG conversion + GNN encoder -------------------------------------------

# These imports are deferred so the extraction layer works in environments that have
# libclang but not torch_geometric.
def _torch():
    import torch
    return torch


def _pyg():
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv
    return Data, GATConv


def graph_to_pyg(
    g: CFunctionGraph, kind_vocab: CursorKindVocab, type_vocab: TypeClassVocab,
    value_vocab: ValueBucketVocab, max_depth: int = 32,
):
    """Convert a CFunctionGraph to a PyG ``Data`` carrying:
      - ``x`` : (N, 4) long tensor [kind_id, type_class_id, value_bucket, clipped_depth]
      - ``edge_index`` : (2, E) long
      - ``edge_type``  : (E,)   long — one of {0=AST_CHILD, 1=DATA_REF, 2=NEXT_STMT}
    """
    torch = _torch()
    Data, _ = _pyg()
    x = torch.tensor([
        [
            kind_vocab.encode(n.kind_name),
            n.type_class,
            n.value_bucket,
            min(n.depth, max_depth - 1),
        ]
        for n in g.nodes
    ], dtype=torch.long)
    if g.edges:
        src = [e.src for e in g.edges]
        dst = [e.dst for e in g.edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor([e.type.value for e in g.edges], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type)


class CEncoder:
    """Lazy torch import — instantiate via ``CEncoder.build(...)``."""

    @staticmethod
    def build(
        kind_vocab: CursorKindVocab,
        type_vocab: TypeClassVocab,
        value_vocab: ValueBucketVocab,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        max_depth: int = 32,
    ):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv

        class _CEncoderModule(nn.Module):
            N_EDGE_TYPES = len(EdgeType)

            def __init__(self) -> None:
                super().__init__()
                self.d_model = d_model
                self.kind_embed = nn.Embedding(kind_vocab.size, d_model)
                self.type_embed = nn.Embedding(type_vocab.size, d_model)
                self.value_embed = nn.Embedding(value_vocab.size, d_model)
                self.depth_embed = nn.Embedding(max_depth, d_model)
                # Per-edge-type GAT layer; a layer = concat(per-edge-type GATs) → linear.
                self.layers = nn.ModuleList()
                for _ in range(n_layers):
                    per_edge = nn.ModuleList([
                        GATConv(d_model, d_model, heads=n_heads, concat=False,
                                add_self_loops=True)
                        for _ in range(self.N_EDGE_TYPES)
                    ])
                    self.layers.append(per_edge)
                self.layer_combine = nn.ModuleList([
                    nn.Linear(d_model * self.N_EDGE_TYPES, d_model) for _ in range(n_layers)
                ])
                # Function-level attention pooling.
                self.pool_query = nn.Parameter(torch.randn(d_model) / d_model**0.5)
                self.norm = nn.LayerNorm(d_model)

            def _seed(self, x: torch.Tensor) -> torch.Tensor:
                # x: (N, 4) long
                e = (
                    self.kind_embed(x[:, 0]) + self.type_embed(x[:, 1])
                    + self.value_embed(x[:, 2]) + self.depth_embed(x[:, 3])
                )
                return e

            def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
                """Returns (ctx: (D,), per_node: (N, D))."""
                h = self._seed(data.x)
                for per_edge, combine in zip(self.layers, self.layer_combine):
                    type_hs = []
                    for t in range(self.N_EDGE_TYPES):
                        mask = data.edge_type == t
                        ei = data.edge_index[:, mask]
                        if ei.numel() == 0:
                            # Pure self-loops fall back to the seed embedding.
                            type_hs.append(h)
                        else:
                            type_hs.append(F.gelu(per_edge[t](h, ei)))
                    h = combine(torch.cat(type_hs, dim=-1))
                    h = self.norm(h)
                # Attention pool: scores = h · pool_query.
                scores = h @ self.pool_query                       # (N,)
                w = torch.softmax(scores, dim=0).unsqueeze(-1)     # (N, 1)
                ctx = (h * w).sum(dim=0)                           # (D,)
                return ctx, h

        return _CEncoderModule()
