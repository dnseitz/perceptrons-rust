
use std::fmt::{self, Debug};
use std::ops::Deref;
use INPUT_SIZE;

#[derive(Debug)]
pub struct Input {
    expected: usize,
    //input: Inner,
    input: Box<[f64]>,
}

struct Inner([f64; INPUT_SIZE]);

impl Input {
    /// Create a new input from a greyscale data format
    ///
    /// The first element should be the expected value and the rest of the data should be in the
    /// range [0-255]
    pub fn from_greyscale(data: &[f64]) -> Self {
        let expected = data[0] as usize;
        let mut input = data.iter().map(|elem| elem / 255.0).collect::<Vec<f64>>().into_boxed_slice();
        input[0] = 1.0;
        Input {
            expected: expected,
            input: input,
        }
    }

    /// Create a new input from a raw data slice
    ///
    /// The first element will be overwritten with 1.0 for the bias value
    pub fn from_raw(expected: usize, data: &[f64]) -> Self {
        let mut input = data.iter().map(f64::clone).collect::<Vec<f64>>().into_boxed_slice();
        input[0] = 1.0;
        Input {
            expected: expected,
            input: input,
        }
    }

    pub fn expected(&self) -> usize {
        self.expected
    }

    pub fn data(&self) -> &[f64] {
        &*self.input
    }

    pub fn iter(&self) -> ::std::slice::Iter<f64> {
        self.input.iter()
    }

    pub fn len(&self) -> usize {
        self.input.len()
    }
}
