# Functional spec: half the sum of squares.
spec sq_sum(x: i32, y: i32) -> r: i32 {
    post: r == ((x * x) + (y * y)) >>u 1
}
