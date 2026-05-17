# Relational spec: integer square root of a non-negative value.
# Any r satisfying r*r <= x < (r+1)*(r+1) is a correct answer.
spec isqrt(x: i32) -> r: i32 {
    pre:  x >=s 0
    post: r * r <=u x  &  (r + 1) * (r + 1) >u x
}
