# Functional spec: signed absolute value, with INT_MIN ruled out by the precondition
# (its absolute value isn't representable as i32). Demonstrates pre-conditions and
# branching written as implications (Select.cond is a *term*, not a Formula).
spec absval(x: i32) -> r: i32 {
    pre:  x != -2147483648:i32
    post: ((x >=s 0) -> (r == x)) & ((x <s 0) -> (r == 0 - x))
}
