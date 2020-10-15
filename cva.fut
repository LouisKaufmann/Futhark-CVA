import "lib/github.com/diku-dk/cpprandom/random"
import "lib/github.com/diku-dk/segmented/segmented"

module dist = normal_distribution f32 minstd_rand

let fixed : f32 = 0.05056643986030719
let r_start : f32 = 0.05

type Swap = {
    term: f32,
    payments : i64,
    notional : f32,
    fixed : f32
}

type Vasicek = {
    a : f32,
    b : f32,
    sigma : f32,
    deltat : f32
}

type Pricing = {
    swap : Swap,
    vasicek: Vasicek,
    t : f32,
    r : f32
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

let gen_remaining (next: f32) (swap_term : f32) (remaining:i64) =
    let seq = map(f32.i64)  (1..2...remaining)
    let remaining_dates = map(+next) ( map(*swap_term) seq)
    in remaining_dates

let swap_price_regular [n] (swap : Swap) (vasicek : Vasicek) (payment_dates:[n]f32) (r:f32) (t:f32) =
    let remaining_dates = payment_dates
    let leg1 = bondprice vasicek r t remaining_dates[0]
    let leg2 = bondprice vasicek r t (remaining_dates[::-1])[0]
    let leg3 = reduce (+) 0 (map (\x -> bondprice vasicek r t x) remaining_dates[1:])
    in swap.notional * (leg1 - leg2 - fixed*swap.term*leg3)

let swapprice (swap : Swap) (vasicek : Vasicek) (r:f32) (t:f32) =
    let payments = f32.ceil (t/swap.term)
    let nextpayment = payments*swap.term
    let remaining = swap.payments - i64.f32 payments
    let remaining_dates = gen_remaining nextpayment swap.term remaining
    let leg1 = bondprice vasicek r t nextpayment
    let leg2 = bondprice vasicek r t (remaining_dates[::-1])[0]
    let leg3 = reduce (+) 0 (map (\x -> bondprice vasicek r t x) remaining_dates)
    in swap.notional * (leg1 - leg2 - fixed*swap.term*leg3)

let gen_payment_dates (swap: Swap) : [] f32 =
    let seq = map(f32.i64) (0..1...swap.payments - 1)
    in map(*swap.term) seq

let gen_times (steps: i64) (years:f32)=
    let sims_per_year = (f32.i64 steps)/years
    let sim_times = map(/sims_per_year) (map(f32.i64) (1..2...steps))
    in sim_times

let shortstep (vasicek : Vasicek) (r:f32) (delta_t:f32) (rand:f32): f32 =
    let delta_r = vasicek.a*(vasicek.b-r) * delta_t + (f32.sqrt delta_t)  * rand * vasicek.sigma
    in r + delta_r

let mc_shortrate (vasicek: Vasicek) (r0:f32) (steps: i64) (rands: [steps] f32): [steps] f32=
    let xs =
        loop x = (replicate steps r0) for i < steps-1 do x with [i+1] = (shortstep vasicek x[i] vasicek.deltat rands[i])
    in xs

let pricing_size (pricing:Pricing) : i64=
    let payments = f32.ceil (pricing.t/pricing.swap.term - 1)
    in pricing.swap.payments - i64.f32 payments

let coef_get (pricing:Pricing) (i : i64) : f32=
    let size = pricing_size(pricing) - 1
    let res = 0
    let res = if i == 0 then res + 1 else res
    let res = if i >0  then res -pricing.swap.fixed*pricing.swap.term else res
    let res = if i == size then res - 1 else res
    in pricing.swap.notional*res

let pricing_get (pricing:Pricing) (i : i64) : f32=
    let start = f32.ceil (pricing.t/pricing.swap.term) * pricing.swap.term
    let coef = (coef_get pricing i)
    let price = bondprice pricing.vasicek pricing.r pricing.t (start + (f32.i64 i)*pricing.swap.term)
    in coef * price

entry main (paths:i64) (steps:i64) (swap_term: f32) (payments: i64) (notional:f32) (a:f32) (b:f32) (sigma: f32) (r0: f32)=
    let rng = minstd_rand.rng_from_seed [1]
    let dt:f32 = (f32.i64 payments * swap_term/ f32.i64 steps)
    let swap : Swap =  {term=swap_term, payments=payments, notional=notional, fixed=fixed}
    let vasicek = {a=a, b=b, sigma=sigma, deltat=dt}
    let duration = f32.i64 swap.payments * swap.term
    let last_date = swap.term * (f32.i64 swap.payments - 1)
    let times = gen_times steps duration
    let rng_vec = minstd_rand.split_rng paths rng
    let payment_amt = swap.payments + 1
    let payment_dates = gen_payment_dates swap
    -- Scenario Generation
    let rands = map (\r ->
        let rng_mat = minstd_rand.split_rng steps r
        let row = map (\x ->
            let (_, v) = dist.rand {mean = 0.0, stddev = 1.0} x
            in v) rng_mat
        in row) rng_vec
    let shortrates = map(\x -> mc_shortrate vasicek r0 steps x) rands

    --Portfolio evaluation for each scenario
    let pricings = map(\x -> map2(\y z ->
                    let pricing : Pricing = {swap = swap, vasicek=vasicek,t = z, r = y}
                    in pricing) x times) shortrates
    let avgexp = map (\x ->
                     let expanded = expand_outer_reduce pricing_size pricing_get (+) 0 x
                     let pfe = map (\y -> f32.max 0 y) expanded
                     in (reduce(+) 0 pfe)/(f32.i64 paths)) (transpose pricings)
    -- let exposures = map(\x ->
    --                     map2 (\y z -> f32.max 0 ( if z > last_date then 0 else swapprice swap vasicek y z)) x times) shortrates
    -- let avgexp = map(\xs -> (reduce(+) 0 xs)/(f32.i64 paths)) (transpose exposures)
    -- Exposure averaging and CVA calculation
    let dexp = map2 (\y z -> y * (bondprice vasicek 0.05 0 z) ) avgexp times
    let CVA = (1-0.4) * 0.01 * reduce (+) 0 (dexp)
    in (CVA, dexp)