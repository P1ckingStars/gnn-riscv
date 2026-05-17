# Relational spec: signed max of two values.
# Any r that is >= both and equals one of them is correct.
spec max(x: i32, y: i32) -> r: i32 {
    post: (r >=s x) & (r >=s y) & ((r == x) | (r == y))
}
