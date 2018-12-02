use log::trace;
use std::f64;
pub fn min_max_norm(array: &[f64], min: f64, max: f64) -> Vec<f64> {
    trace!("Min {} Max {}", min, max);
    let min_a = array.iter().fold(f64::NAN, |acc, &v| f64::min(acc, v));
    let max_a = array.iter().fold(f64::NAN, |acc, &v| f64::max(acc, v));
    trace!("Min_a {} Max_a {} ", min_a, max_a);
    // Standard Normalization
    let cur_diff = max_a - min_a;
    let new_diff = max - min;
    trace!("Cur_diff {} New_diff {} ", cur_diff, new_diff);
    array
        .iter()
        .map(|v| {
            let scaled = (v - min_a) / cur_diff * new_diff + min;
            trace!("v {} scaled {} ", v, scaled);
            scaled
        })
        .collect::<Vec<_>>()
}

pub fn mean(array: &[f64]) -> f64 {
    let sum: f64 = array.iter().map(|&v| v).sum();
    sum / array.len() as f64
}

pub fn std_dev(array: &[f64]) -> (f64, f64) {
    let mean = mean(array);
    let sum_squares: f64 = array.iter().map(|v| (v - mean) * (v - mean)).sum();
    (mean, (sum_squares / array.len() as f64).sqrt())
}
pub fn tanh_est(array: &[f64]) -> Vec<f64> {
    let (mean, std_dev) = std_dev(array);
    trace!("TanH estimator Mean:{} StdDev: {}", mean, std_dev);
    array
        .iter()
        .map(|v| {
            let scaled = ((0.01 * ((v - mean) / std_dev)).tanh() + 1.0) * 0.5;
            trace!("v {} scaled {} ", v, scaled);
            scaled
        })
        .collect::<Vec<_>>()
}
