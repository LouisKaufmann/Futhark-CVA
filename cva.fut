import "lib/github.com/diku-dk/cpprandom/random"
import "lib/github.com/diku-dk/segmented/segmented"

module dist = normal_distribution f32 minstd_rand

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
    deltat : f32,
    r0 : f32
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

let gen_remaining (next: f32) (swap_term : f32) (remaining:i64) =
    let seq = map(f32.i64)  (1..2...remaining)
    let remaining_dates = map(+next) ( map(*swap_term) seq)
    in remaining_dates

let swapprice (swap : Swap) (vasicek : Vasicek) (r:f32) (t:f32) =
    let payments = f32.ceil (t/swap.term)
    let nextpayment = payments*swap.term
    let remaining = swap.payments - i64.f32 payments
    let remaining_dates = gen_remaining nextpayment swap.term remaining
    let leg1 = bondprice vasicek r t nextpayment
    let leg2 = bondprice vasicek r t (remaining_dates[::-1])[0]
    let leg3 = reduce (+) 0 (map (\x -> bondprice vasicek r t x) remaining_dates)
    in swap.notional * (leg1 - leg2 - swap.fixed*swap.term*leg3)

let gen_payment_dates (swap_payments: i64) (swap_term:f32) =
    let seq = map(f32.i64) (1..2...swap_payments)
    in map(*swap_term) seq


let gen_times (steps: i64) (years:f32)=
    let sims_per_year = (f32.i64 steps)/years
    let sim_times = map(/sims_per_year) (map(f32.i64) (1..2...steps))
    in sim_times

--Scalar function for single step in vasicek shirt
let shortstep (vasicek : Vasicek) (r:f32) (rand:f32): f32 =
    let delta_r = vasicek.a*(vasicek.b-r) * vasicek.deltat + (f32.sqrt vasicek.deltat)  * rand * vasicek.sigma
    in r + delta_r

--Sequential Monte carlo shortrate path simulation
let mc_shortrate (vasicek: Vasicek) (r0:f32) (steps: i64) (rands: [steps] f32): [steps] f32=
    let xs =
        loop x = (replicate steps r0) for i < steps-1 do x with [i+1] = (shortstep vasicek x[i] rands[i])
    in xs

--Get and size functions needed for expand_reduce function in
let pricing_size (pricing:Pricing) : i64=
    let payments = f32.ceil (pricing.t/pricing.swap.term - 1)
    in i64.max (pricing.swap.payments - i64.f32 payments) 0

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
    in coef * price * pricing.swap.notional

-- Set the fixed rate, given a swap term, number of payments and a vasicek model
let set_fixed_rate (swap_term: f32) (swap_payments: i64) (vasicek:Vasicek) : f32 =
    let payment_dates = gen_payment_dates swap_payments swap_term
    let leg1 = bondprice vasicek vasicek.r0 0 (payment_dates[::-1])[0]
    let sum = reduce (+) 0 (map (\x -> bondprice vasicek vasicek.r0 0 x) payment_dates)
    in (1.0 - leg1)/(sum*swap_term)


entry main [n]  (paths:i64) (steps:i64) (swap_term: [n]f32) (payments: [n]i64)
                (notional:[n]f32) (a:f32) (b:f32) (sigma: f32) (r0: f32)=
    let max_duration = f32.maximum (map2 (\x y -> x * (f32.i64 y)) swap_term payments)
    let dt:f32 = (max_duration/ f32.i64 steps)
    let vasicek = {a=a, b=b, sigma=sigma, deltat=dt,r0 = r0}
    let swaps = map (\x : Swap ->
        {term=swap_term[x],
        payments=payments[x],
        notional=notional[x],
        fixed=(set_fixed_rate swap_term[x] payments[x] vasicek)}) (indices swap_term)

    let times = gen_times steps max_duration

    let rng = minstd_rand.rng_from_seed [1]
    let rng_vec = minstd_rand.split_rng paths rng
    -- -- Scenario Generation
    let rands = map (\r ->
        let rng_mat = minstd_rand.split_rng steps r
        let row = map (\x ->
            let (_, v) = dist.rand {mean = 0.0, stddev = 1.0} x
            in v) rng_mat
        in row) rng_vec
    let shortrates = map(\x -> mc_shortrate vasicek r0 steps x) rands
    -- --Portfolio evaluation for each scenario

    let prices : [steps] [paths] [n] f32 = map2(\x z->
                        let pricings = map(\y ->
                            map(\swap ->
                                let pricing : Pricing = {swap = swap, vasicek=vasicek,t = z, r = y}
                                in pricing
                            ) swaps) x
                        let flattened = flatten pricings
                        let prices = expand_outer_reduce pricing_size pricing_get (+) 0 flattened
                        let unflattened : [paths] [n] f32 = unflatten paths n prices
                        in unflattened
                    ) (transpose shortrates) times
    let avgexp = map (\(xs : [paths] [n] f32) : f32->
                    let netted : [paths] f32 = map(\x-> reduce (+) 0 x) (xs)
                    let pfe = map (\x -> f32.max 0 x) netted
                    in (reduce(+) 0 pfe)/(f32.i64 paths)
                ) (prices)

    -- -- Exposure averaging and CVA calculation
    let dexp = map2 (\y z -> y * (bondprice vasicek 0.05 0 z) ) avgexp times
    let CVA = (1-0.4) * 0.01 * reduce (+) 0 (dexp)
    in (CVA, avgexp)

-- ==
-- entry: test
-- input {  1000i64 100i64 }
-- input { 100000i64 500i64 }
-- input { 1000000i64 1500i64 }

entry test (paths:i64) (steps:i64) : f32 =
  main paths steps [1,0.5,0.25,0.1,0.3,0.1,2,3,1,1,0.5,0.25,0.1,0.3,0.1,2,3,1,1,0.5,0.25,0.1,0.3,0.1,2,3,1,1,0.5,0.25,0.1,0.3,0.1,2,3,1,1,0.5,0.25,0.1,0.3,0.1,2,3,1]
  [10,20,5,5,50,20,30,15,18,10,200,5,5,50,20,30,15,18,10,20,5,5,100,20,30,15,18,10,20,5,5,50,20,30,15,18,10,20,5,5,50,20,30,15,18]
  [1,-0.5,1,1,1,1,1,1,1,1,-0.5,1,1,1,1,1,1,1,1,-0.5,1,1,1,1,1,1,1,1,-0.5,1,1,1,1,1,1,1,1,-0.5,1,1,1,1,1,1,1] 0.01 0.05 0.001 0.05 |> (.0)
