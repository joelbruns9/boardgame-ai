//! Isolated review microbenchmark, not a production encoder benchmark.
//! rustc -O reviews/welcome_to_draw_kernel_bench.rs -o <temporary exe>
use std::{hint::black_box, time::Instant};
const K: usize = 15;
struct Case { counts: [f64; K], masks: [[f64; K]; 3], num: Vec<f64>, den: f64 }
fn joint(c: &Case) -> f64 {
    // Mirrors encoder.rs DrawCounts::probability's 0/1-mask loop.
    let mut hits = 0.0;
    for a in 0..K {
        if c.masks[0][a] == 0.0 { continue; }
        for b in 0..K {
            if c.masks[1][b] == 0.0 { continue; }
            let base = (a * K + b) * K;
            for z in 0..K {
                if c.masks[2][z] != 0.0 { hits += c.num[base + z]; }
            }
        }
    }
    (hits / c.den.max(1e-6)).max(0.0).min(1.0)
}
fn closed(c: &Case) -> f64 {
    let (mut s0,mut s1,mut s2,mut p01,mut p02,mut p12,mut t) = (0.,0.,0.,0.,0.,0.,0.);
    for i in 0..K {
        let n = c.counts[i];
        let (a,b,z) = (c.masks[0][i],c.masks[1][i],c.masks[2][i]);
        s0+=n*a; s1+=n*b; s2+=n*z;
        p01+=n*a*b; p02+=n*a*z; p12+=n*b*z; t+=n*a*b*z;
    }
    let hits = s0*s1*s2 - p01*s2 - p02*s1 - p12*s0 + 2.*t;
    (hits / c.den.max(1e-6)).max(0.0).min(1.0)
}
fn main() {
    let mut seed = 20260925u64;
    let mut next = || {seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1); seed>>32};
    let mut cases = Vec::new();
    for _ in 0..256 {
        let counts: [f64; K] = std::array::from_fn(|_| (next()%6) as f64);
        let masks = std::array::from_fn(|_| std::array::from_fn(|_| (next()%2) as f64));
        let total: f64 = counts.iter().sum();
        let mut num = Vec::new();
        for a in 0..K { for b in 0..K { for z in 0..K {
            num.push((counts[a]*(counts[b]-f64::from(a==b))
                *(counts[z]-f64::from(a==z)-f64::from(b==z))).max(0.));
        } } }
        let c = Case {counts, masks, num, den: total*(total-1.)*(total-2.)};
        assert_eq!(joint(&c).to_bits(), closed(&c).to_bits());
        cases.push(c);
    }
    let iterations = 200_000;
    for (label, f) in [("joint", joint as fn(&Case)->f64), ("closed", closed as fn(&Case)->f64)] {
        let start = Instant::now();
        let mut checksum = 0.;
        for i in 0..iterations {checksum += black_box(f(black_box(&cases[i%cases.len()])));}
        println!("{label}: {:.1} ns/call; checksum {checksum}", start.elapsed().as_nanos() as f64 / iterations as f64);
    }
    println!("256 bit-exact comparisons; isolated precomputed-joint kernel, synthetic binary masks");
}
