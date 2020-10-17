import "lib/github.com/diku-dk/cpprandom/random"
import "lib/github.com/diku-dk/segmented/segmented"

module dist = normal_distribution f32 minstd_rand

type Swap = {
    term: f32,
    payments : i64,
    notional : f32,
    fixed : f32,
    netting : i64
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
    let seq = map(f32.i64) (0..1...swap_payments - 1)
    in map(*swap_term) seq


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
    in coef * price

let set_fixed_rate (swap_term: f32) (swap_payments: i64) (vasicek:Vasicek) : f32 =
    let payment_dates = gen_payment_dates swap_payments swap_term
    let leg1 = bondprice vasicek vasicek.r0 0 (payment_dates[::-1])[0]
    let sum = reduce (+) 0 (map (\x -> bondprice vasicek vasicek.r0 0 x) payment_dates[1:])
    in (1.0 - leg1)/(sum*swap_term)

entry main [n]  (paths:i64) (steps:i64) (netting: [n]i64) (swap_term: [n]f32) (payments: [n]i64)
                (notional:[n]f32) (a:f32) (b:f32) (sigma: f32) (r0: f32)=
    let max_duration = f32.maximum (map2 (\x y -> x * (f32.i64 y)) swap_term payments)
    let dt:f32 = (max_duration/ f32.i64 steps)
    let vasicek = {a=a, b=b, sigma=sigma, deltat=dt,r0 = r0}
    let swaps = map (\x : Swap ->
        {term=swap_term[x],
        payments=payments[x],
        notional=notional[x],
        netting=netting[x],
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

    -- -- Three approaches
    -- -- First approach - create a pricings 2d array before, and map over
    -- -- let pricings = map(\x -> map2(\y z ->
    -- --                 let pricing : Pricing = {swap = swap, vasicek=vasicek,t = z, r = y}
    -- --                 in pricing) x times) shortrates
    -- -- let avgexp = map (\x ->
    -- --                  let expanded = expand_reduce pricing_size pricing_get (+) 0 x :> [paths] f32
    -- --                  let pfe = map (\y -> f32.max 0 y) expanded
    -- --                  in (reduce(+) 0 pfe)/(f32.i64 paths) (transpose pricings)

    -- -- Second approach - Create pricing array within map, to reduce intermediate arrays
    let avgexp = map2 (\xs time ->
                    let pfes = map(\swap ->
                        let pricings: [] Pricing = map(\x-> {swap = swap, vasicek=vasicek,t = time, r = x}) xs
                        let expanded = expand_reduce pricing_size pricing_get (+) 0 pricings
                        let pfe = map (\y -> f32.max 0 y) expanded
                        in pfe
                    ) swaps
                    let netting = map2 (\x y-> map(\z -> z*(f32.i64 y.netting)) x) pfes swaps
                    let netted = map(\x -> reduce (+) 0 x) (transpose netting)
                    in (reduce(+) 0 netted)/(f32.i64 paths)
                ) (transpose shortrates) times

    -- -- Third approach - dynamic memory (probably sequential execution)
    -- -- let exposures = map(\x ->
    -- --                     map2 (\y z -> f32.max 0 ( if z > last_date then 0 else swapprice swap vasicek y z)) x times) shortrates
    -- -- let avgexp = map(\xs -> (reduce(+) 0 xs)/(f32.i64 paths)) (transpose exposures)


    -- -- Exposure averaging and CVA calculation
    let dexp = map2 (\y z -> y * (bondprice vasicek 0.05 0 z) ) avgexp times
    let CVA = (1-0.4) * 0.01 * reduce (+) 0 (dexp)
    in (CVA, dexp)