# Relational spec: absolute difference. Two correct outputs in i8 (x-y or y-x);
# the smaller pre/post would be functional with abs() outside, but in a bitvector
# setting the natural relational form admits both.
spec absdiff(x: i8, y: i8) -> r: i8 {
    post: (r == x - y) | (r == y - x)
}
