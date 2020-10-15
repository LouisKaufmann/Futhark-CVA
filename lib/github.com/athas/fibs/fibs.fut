-- | Generating the Fibonacci sequence: *1, 1, 2, 3, 5, 8, 13*, and so
-- forth.
--
-- This file provides both parallel and sequential implementations.
-- In practice, the sequential implementations are probably preferable
-- in all realistic cases, as the integers storing the Fibonacci
-- numbers will overflow far before enough parallelism is available.

local type Q = (i32,i32,i32,i32)

local let mul (x00,x01,x10,x11) (y00,y01,y10,y11) : Q  =
  (x00*y00+x01*y10,
   x00*y01+x01*y11,
   x10*y00+x11*y10,
   x10*y01+x11*y11)

local let Q : Q = (1,1,1,0)
local let ne : Q = (1,0,0,1)

-- | Generate the first `n` Fibonacci numbers using a parallel
-- algorithm.
--
-- **Work:** *O(n)*
--
-- **Span:** *O(log(n))*
let fibs (n: i64): [n]i32 = replicate n Q |> scan mul ne |> map (.1)

-- | Generate the `n`th Fibonacci number (0-indexed) using a parallel
-- algorithm.  Specifically, `fib n == last (fibs (n+1)`, but `fib` is
-- more efficient.  However, `fib_seq`@term is likely even more
-- efficient.
--
-- **Work:** *O(n)*
--
-- **Span:** *O(log(n))*
let fib (n: i64) = replicate n Q |> reduce mul ne |> (.0)

-- | Generate the first `n` Fibonacci numbers using a sequential
-- *O(n)* algorithm.
let fibs_seq (n: i64): [n]i32 =
  loop arr = replicate n 1 for i in 2..<n do
    let x = arr[i-1]
    let y = arr[i-2]
    let arr[i] = x + y
    in arr

-- | Generate the `n`th Fibonacci number (0-indexed) using a
-- sequential *O(n)* algorithm.
let fib_seq(n: i64): i32 =
  let (x,_) = loop (x, y) = (1,1) for _i < n do (y, x+y)
  in x
