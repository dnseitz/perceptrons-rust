
use INPUT_SIZE;
use rand::{self, Rng};
use std::fmt::{self, Debug};
use input::Input;

pub struct Perceptron {
    weights: [f64; INPUT_SIZE],
    target_class: u32,
}

impl Perceptron {
    pub fn new(target_class: u32) -> Self {
        let mut weights = [0f64; INPUT_SIZE];
        for weight in &mut weights[..] {
            *weight = rand::thread_rng().gen_range(-0.05f64, 0.05f64);
        }
        Perceptron {
            weights: weights,
            target_class: target_class,
        }
    }

    pub fn target_class(&self) -> u32 {
        self.target_class
    }

    pub fn calculate(&self, input: &Input) -> f64 {
        self.weights.iter().zip(input.iter()).map(|(w, i)| w * i).sum()
    }

    pub fn update(&mut self, learning_rate: f64, input: &Input) {
        let result = if self.calculate(input) > 0f64 { 1 } else { 0 };
        let expected = if self.target_class == input.expected() { 1 } else { 0 };

        if result != expected {
            for (weight, input) in self.weights[..].iter_mut().zip(input.iter()) {
                *weight += learning_rate * ((expected - result) as f64) * input;
            }
        }
    }
}

impl Debug for Perceptron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let _ = write!(f, "Perceptron {{\nWeights: [{}", self.weights[0])?;
        for elem in self.weights[1..].iter() {
            let _ = write!(f, ", {}", elem)?;
        }
        write!(f, "]\n}}")
    }
}
