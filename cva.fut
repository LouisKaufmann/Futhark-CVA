    import "lib/github.com/diku-dk/cpprandom/random"
    module dist = normal_distribution f32 minstd_rand


let fixed : f32 = 0.05056643986030719
let r_start : f32 = 0.05

type Swap = {
    term: f32,
    payments : i32,
    notional : f32
}

type Vasicek = {
    a : f32,
    b : f32,
    sigma : f32,
    deltat : f32
}

let exp (a:f32) : f32 =
    f32.e ** (a)

let bondprice (vasicek: Vasicek) (r:f32) (t:f32) (T:f32) : f32 =
    let B = (1 - exp (-vasicek.a*(T-t))) / vasicek.a
    let A1 = (B - T + t) * (vasicek.a ** 2 * vasicek.b - (vasicek.sigma ** 2)/2)/(vasicek.a**2)
    let A2 = (vasicek.sigma ** 2 * B**2) / (4*vasicek.a)
    let A = exp (A1 - A2)
    in A * exp (-B*r)

let bond_price (a: f32) (b: f32) (sigma: f32) (r:f32) (t:f32) (T:f32) : f32 =
    let B = (1 - exp (-a*(T-t))) / a
    let A1 = (B - T + t) * (a ** 2 * b - (sigma ** 2)/2)/(a**2)
    let A2 = (sigma ** 2 * B**2) / (4*a)
    let A = exp (A1 - A2)
    in A * exp (-B*r)

let gen_remaining (next: f32) (swap_term : f32) (remaining:i32) =
    let seq = map(f32.i32)  (1..2...remaining)
    let remaining_dates = map(+next) ( map(*swap_term) seq)
    in remaining_dates :> [remaining] f32

let swapprice (swap : Swap) (vasicek : Vasicek) (r:f32) (t:f32) =
    let payments = f32.ceil (t/swap.term)
    let nextpayment = payments*swap.term
    --let mask = filter (>nextpayment) payments_arr
    --let fixed_flows = map (\x -> bondprice vasicek r t x) mask
    --let a = replicate 100 10
    let leg1 = bondprice vasicek r t nextpayment
    let leg2 = bondprice vasicek r t 20 -- (payments_arr[::-1])[0]
    let leg3 = reduce (+)
    --let leg3 = reduce (+) 0 (map(\x-> bondprice vasicek r t x) (a))
    --let leg3 = reduce (+) 0 (fixed_flows)
    in swap.notional * (leg1 - leg2)-- fixed*swap.term*leg3)

-- let swapprice (swap : Swap) (vasicek : Vasicek) (payments_arr: []f32) (r:f32) (t:f32) =
--     let payments = f32.ceil (t/swap.term)
--     let nextpayment = payments*swap.term
--     let remaining = swap.payments - i32.f32 payments
--     let remaining_dates = gen_remaining nextpayment swap.term remaining
--     let leg1 = bondprice vasicek r t nextpayment
--     let leg2 = bondprice vasicek r t (remaining_dates[::-1])[0]
--     let leg3 = reduce (+) 0 (map (\x -> bondprice vasicek r t x) remaining_dates)
--     in swap.notional * (leg1 - leg2 - fixed*swap.term*leg3)

let gen_payment_dates (swap: Swap) =
    let seq = map(f32.i32) (0..1...swap.payments)
    in map(*swap.term) seq :> [] f32

let gen_times (steps: i32) (years:f32)=
    let sims_per_year = (f32.i32 steps)/years
    let sim_times = map(/sims_per_year) (map(f32.i32) (1..2...steps))
    in sim_times

let shortstep (vasicek : Vasicek) (r:f32) (delta_t:f32) (rand:f32): f32 =
    let delta_r = vasicek.a*(vasicek.b-r) * delta_t + (f32.sqrt delta_t)  * rand * vasicek.sigma
    in r + delta_r

let mc_shortrate (vasicek: Vasicek) (r0:f32) (steps: i32) (rands: [steps] f32): [steps] f32=
    let xs =
        loop x = (replicate steps r0) for i < steps-1 do x with [i+1] = (shortstep vasicek x[i] vasicek.deltat rands[i]) --++ [(shortstep vasicek x[i] delta_t rands[i])]
    in xs :> [steps] f32

let numerical_integrate (xs: [] f32) (delta: f32)=
    let xs = xs ++ [0]
    let xs2 = rotate (-1) xs
    in delta/2 * (reduce(+) 0
                (map2(\x y -> x + y) xs xs2))


let main (paths:i32) (steps:i32) (swap_term: f32) (payments: i32) (notional:f32) (a:f32) (b:f32) (sigma: f32) (r0: f32)=
    let rng = minstd_rand.rng_from_seed [1]
    let dt:f32 = (f32.i32 payments * swap_term/ f32.i32 steps)
    let swap : Swap =  {term=swap_term, payments=payments, notional=notional}
    let vasicek = {a=a, b=b, sigma=sigma, deltat=dt}
    let duration = f32.i32 swap.payments * swap.term
    let last_date = swap.term * (f32.i32 swap.payments - 1)
    let times = gen_times steps duration
    let rng_vec = minstd_rand.split_rng paths rng
    let rands = map (\r ->
        let rng_mat = minstd_rand.split_rng steps r
        let row = map (\x ->
            let (_, v) = dist.rand {mean = 0.0, stddev = 1.0} x
            in v) rng_mat
        in row) rng_vec
    let payment_amt = swap.payments + 1
    let payment_dates = gen_payment_dates swap :> [payment_amt] f32
    -- let mcs =
    --     loop xs = [] for i < paths do
    --         xs ++ [(mc_shortrate vasicek r0 steps delta_t rands[i])]
    let mcs = map(\x -> mc_shortrate vasicek r0 steps x) rands
    let exposures = map(\x ->
                    let exp  = map (\y -> if y < 0 then 0 else y)
                        (map2 (\y z -> if z < last_date then swapprice swap vasicek y z else 0) x times) --1 else 0) x times) 
                    in exp) mcs
    let avgexp = map(\xs -> (reduce(+) 0 xs)/(f32.i32 paths)) (transpose exposures) :> [steps] f32
    let dexp = map2 (\y z -> y * (bondprice vasicek 0.05 0 z) ) avgexp times
    let CVA = (1-0.4) * 0.01 * reduce (+) 0 (dexp)
    in (dexp, CVA)