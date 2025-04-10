pub trait ActivationFunction {
    fn function(x: f64) -> f64;
    fn derivative(x: f64) -> f64;
}

pub struct Linear;
impl ActivationFunction for Linear {
    fn function(x: f64) -> f64 {
        x
    }
    fn derivative(_x: f64) -> f64 {
        1.0
    }
}

pub struct ReLU;
impl ActivationFunction for ReLU {
    fn function(x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }
    fn derivative(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}
