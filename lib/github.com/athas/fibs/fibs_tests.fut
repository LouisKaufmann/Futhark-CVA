-- | ignore

import "fibs"

-- ==
-- entry: test_fibs
-- input { 10 } output { [1,1,2,3,5,8,13,21,34,55] }
entry test_fibs = fibs

-- ==
-- entry: test_fib
-- input { 9 } output { 55 }
entry test_fib = fib

-- ==
-- entry: test_fibs_seq
-- input { 10 } output { [1,1,2,3,5,8,13,21,34,55] }
entry test_fibs_seq = fibs_seq

-- ==
-- entry: test_fib_seq
-- input { 9 } output { 55 }
entry test_fib_seq = fib_seq
