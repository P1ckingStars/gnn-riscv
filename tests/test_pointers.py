"""Tests for the memory + pointer extension.

Covers the DSL layer (PtrTy, PtrAdd, Load, type-check), the interpreter (Memory,
byte-level read), the serializer round-trip, and Z3 spec-vs-spec equivalence with
memory loads.

What's NOT covered here (deferred work):
  - Parser surface syntax for ``*p``, ``a[i]`` — defer to a parser extension task.
  - Asm-side load/store in ``_exec_asm`` — the SMT verifier handles memory on the
    spec side only for now.
"""
from __future__ import annotations

import pytest

from spec.dsl import (
    Bin, BinOp, BoolTrue, Cmp, CmpOp, Const, Load, Param, PtrAdd, PtrTy, Spec, Ty,
    Var, type_check,
)
from spec.interpreter import Memory, evaluate, satisfies
from spec.serialize import from_dict, to_dict


# ---- Type-system basics -----------------------------------------------------

def test_ptr_ty_basics():
    t = PtrTy(elem_ty=Ty.I32)
    assert t.width == 64
    assert t.mask == 0xFFFF_FFFF_FFFF_FFFF
    assert t.elem_size_bytes == 4
    assert "I32" in repr(t)
    # Nested pointer
    pp = PtrTy(elem_ty=PtrTy(elem_ty=Ty.I64))
    assert pp.elem_size_bytes == 8   # pointers are XLEN/8 bytes


def test_type_check_accepts_pointer_load_spec():
    """spec(p: *i32) -> r: i32 { post: r == *p }"""
    p_param = Param("p", PtrTy(Ty.I32))
    r_param = Param("r", Ty.I32)
    p_var = Var(ty=PtrTy(Ty.I32), name="p")
    r_var = Var(ty=Ty.I32, name="r")
    load = Load(ty=Ty.I32, ptr=p_var)
    spec = Spec(
        inputs=(p_param,), outputs=(r_param,),
        post=Cmp(op=CmpOp.EQ, lhs=r_var, rhs=load),
    )
    type_check(spec)  # must not raise


def test_type_check_rejects_load_on_scalar():
    """``Load(ptr=Var(I32, x))`` is a type error — Load requires a pointer."""
    x_var = Var(ty=Ty.I32, name="x")
    bad = Load(ty=Ty.I32, ptr=x_var)
    r_var = Var(ty=Ty.I32, name="r")
    spec = Spec(
        inputs=(Param("x", Ty.I32),), outputs=(Param("r", Ty.I32),),
        post=Cmp(op=CmpOp.EQ, lhs=r_var, rhs=bad),
    )
    with pytest.raises(TypeError, match="Load"):
        type_check(spec)


def test_type_check_rejects_bin_on_pointers():
    """``ptr + ptr`` via Bin is wrong — must use PtrAdd. (Even if the result type
    happens to be PtrTy, Bin's scalar-only operand rule fires first.)"""
    p_var = Var(ty=PtrTy(Ty.I32), name="p")
    q_var = Var(ty=PtrTy(Ty.I32), name="q")
    bad = Bin(ty=PtrTy(Ty.I32), op=BinOp.ADD, lhs=p_var, rhs=q_var)
    r_var = Var(ty=PtrTy(Ty.I32), name="r")
    spec = Spec(
        inputs=(Param("p", PtrTy(Ty.I32)), Param("q", PtrTy(Ty.I32))),
        outputs=(Param("r", PtrTy(Ty.I32)),),
        post=Cmp(op=CmpOp.EQ, lhs=r_var, rhs=bad),
    )
    with pytest.raises(TypeError, match="pointer"):
        type_check(spec)


def test_type_check_rejects_pointer_signed_cmp():
    p_var = Var(ty=PtrTy(Ty.I32), name="p")
    q_var = Var(ty=PtrTy(Ty.I32), name="q")
    spec = Spec(
        inputs=(Param("p", PtrTy(Ty.I32)), Param("q", PtrTy(Ty.I32))),
        outputs=(Param("r", Ty.I32),),
        post=Cmp(op=CmpOp.SLT, lhs=p_var, rhs=q_var),
    )
    with pytest.raises(TypeError, match="pointers"):
        type_check(spec)


def test_type_check_allows_pointer_eq():
    p_var = Var(ty=PtrTy(Ty.I32), name="p")
    q_var = Var(ty=PtrTy(Ty.I32), name="q")
    spec = Spec(
        inputs=(Param("p", PtrTy(Ty.I32)), Param("q", PtrTy(Ty.I32))),
        outputs=(Param("r", Ty.I32),),
        # r == 1 iff p and q are the same address
        post=Cmp(op=CmpOp.EQ, lhs=p_var, rhs=q_var),
    )
    type_check(spec)


# ---- Memory + interpreter ---------------------------------------------------

def test_memory_from_arrays_round_trips_each_element():
    m = Memory.from_arrays({1000: [7, 11, -3, 13]}, elem_ty=Ty.I32)
    assert m.load(1000, Ty.I32) == 7
    assert m.load(1004, Ty.I32) == 11
    assert m.load(1008, Ty.I32) == 0xFFFFFFFD   # -3 in unsigned canonical
    assert m.load(1012, Ty.I32) == 13


def test_memory_load_unmapped_bytes_returns_zero():
    m = Memory()
    assert m.load(9999, Ty.I32) == 0


def test_memory_store_then_load_round_trip_i64():
    m = Memory()
    m.store(2000, 0x1234_5678_9ABC_DEF0, Ty.I64)
    assert m.load(2000, Ty.I64) == 0x1234_5678_9ABC_DEF0


def test_interpreter_load_through_pointer():
    """spec(p: *i32) -> r: i32 { post: r == *p } evaluated against a known buffer."""
    p_var = Var(ty=PtrTy(Ty.I32), name="p")
    spec = Spec.functional(
        inputs=(Param("p", PtrTy(Ty.I32)),),
        body=Load(ty=Ty.I32, ptr=p_var),
    )
    mem = Memory.from_arrays({1000: [42]}, elem_ty=Ty.I32)
    assert evaluate(spec, (1000,), memory=mem) == 42


def test_interpreter_ptradd_indexes_into_array():
    p_var = Var(ty=PtrTy(Ty.I32), name="p")
    # Read p[2] = *(p + 8)
    body = Load(ty=Ty.I32, ptr=PtrAdd(ty=PtrTy(Ty.I32), base=p_var,
                                       offset=Const(ty=Ty.I32, value=8)))
    spec = Spec.functional(inputs=(Param("p", PtrTy(Ty.I32)),), body=body)
    mem = Memory.from_arrays({1000: [10, 20, 30, 40]}, elem_ty=Ty.I32)
    assert evaluate(spec, (1000,), memory=mem) == 30


def test_interpreter_array_sum_3():
    """Functional ``sum3``: r == a[0] + a[1] + a[2]. Pure-load straight-line code."""
    a = Var(ty=PtrTy(Ty.I32), name="a")
    l0 = Load(ty=Ty.I32, ptr=a)
    l1 = Load(ty=Ty.I32, ptr=PtrAdd(ty=PtrTy(Ty.I32), base=a,
                                     offset=Const(ty=Ty.I32, value=4)))
    l2 = Load(ty=Ty.I32, ptr=PtrAdd(ty=PtrTy(Ty.I32), base=a,
                                     offset=Const(ty=Ty.I32, value=8)))
    body = Bin(ty=Ty.I32, op=BinOp.ADD,
               lhs=Bin(ty=Ty.I32, op=BinOp.ADD, lhs=l0, rhs=l1),
               rhs=l2)
    spec = Spec.functional(inputs=(Param("a", PtrTy(Ty.I32)),), body=body)
    mem = Memory.from_arrays({500: [100, 7, 3, 999]}, elem_ty=Ty.I32)
    assert evaluate(spec, (500,), memory=mem) == 110


def test_satisfies_works_with_pointer_input_and_memory():
    a = Var(ty=PtrTy(Ty.I32), name="a")
    r = Var(ty=Ty.I32, name="r")
    # Postcondition: r equals *(a + 4)
    post = Cmp(op=CmpOp.EQ, lhs=r,
               rhs=Load(ty=Ty.I32, ptr=PtrAdd(ty=PtrTy(Ty.I32), base=a,
                                               offset=Const(ty=Ty.I32, value=4))))
    spec = Spec(inputs=(Param("a", PtrTy(Ty.I32)),),
                outputs=(Param("r", Ty.I32),), post=post)
    mem = Memory.from_arrays({2000: [0xdead, 0xbeef]}, elem_ty=Ty.I32)
    assert satisfies(spec, (2000,), (0xbeef,), memory=mem)
    assert not satisfies(spec, (2000,), (0xdead,), memory=mem)


# ---- Serializer round-trip --------------------------------------------------

def test_serialize_roundtrip_pointer_spec():
    a = Var(ty=PtrTy(Ty.I32), name="a")
    body = Load(ty=Ty.I32, ptr=PtrAdd(ty=PtrTy(Ty.I32), base=a,
                                       offset=Const(ty=Ty.I32, value=4)))
    spec = Spec.functional(inputs=(Param("a", PtrTy(Ty.I32)),), body=body)
    spec2 = from_dict(to_dict(spec))
    assert spec == spec2
    type_check(spec2)


def test_serialize_roundtrip_nested_pointer_type():
    pp_ty = PtrTy(elem_ty=PtrTy(elem_ty=Ty.I64))
    spec = Spec(
        inputs=(Param("pp", pp_ty),),
        outputs=(Param("r", Ty.I64),),
        post=Cmp(op=CmpOp.EQ,
                 lhs=Var(ty=Ty.I64, name="r"),
                 rhs=Load(ty=Ty.I64,
                           ptr=Load(ty=PtrTy(elem_ty=Ty.I64),
                                    ptr=Var(ty=pp_ty, name="pp")))),
    )
    spec2 = from_dict(to_dict(spec))
    assert spec == spec2
    type_check(spec2)


# ---- SMT spec-vs-spec equivalence with memory -------------------------------

def test_smt_proves_load_through_ptradd_equals_direct_index():
    """*(p + 0) ≡ *p — sanity that PtrAdd(p, 0) is a no-op."""
    from eval.verify_smt import prove_functional_equiv
    p_var = Var(ty=PtrTy(Ty.I32), name="p")
    spec_a = Spec.functional(
        inputs=(Param("p", PtrTy(Ty.I32)),),
        body=Load(ty=Ty.I32, ptr=p_var),
    )
    spec_b = Spec.functional(
        inputs=(Param("p", PtrTy(Ty.I32)),),
        body=Load(ty=Ty.I32, ptr=PtrAdd(ty=PtrTy(Ty.I32), base=p_var,
                                         offset=Const(ty=Ty.I32, value=0))),
    )
    v = prove_functional_equiv(spec_a, spec_b, timeout_s=5.0)
    assert v.equivalent, f"counterexample: {v.counterexample}"


def test_smt_finds_counterexample_for_non_equiv_specs():
    from eval.verify_smt import prove_functional_equiv
    p_var = Var(ty=PtrTy(Ty.I32), name="p")
    # spec A: *p
    spec_a = Spec.functional(
        inputs=(Param("p", PtrTy(Ty.I32)),),
        body=Load(ty=Ty.I32, ptr=p_var),
    )
    # spec B: *(p + 4)  — different address
    spec_b = Spec.functional(
        inputs=(Param("p", PtrTy(Ty.I32)),),
        body=Load(ty=Ty.I32, ptr=PtrAdd(ty=PtrTy(Ty.I32), base=p_var,
                                         offset=Const(ty=Ty.I32, value=4))),
    )
    v = prove_functional_equiv(spec_a, spec_b, timeout_s=5.0)
    assert not v.equivalent
    assert v.counterexample is not None
