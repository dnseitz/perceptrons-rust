
#[derive(Debug)]
pub struct Input {
    expected: usize,
    input: Box<[f64]>,
}

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
    #[cfg(test)]
    pub fn from_raw(expected: usize, data: &[f64]) -> Self {
        let mut input = data.iter().map(f64::clone).collect::<Vec<f64>>().into_boxed_slice();
        input[0] = 1.0;
        Input {
            expected: expected,
            input: input,
        }
    }

    /// Get the expected value for this input
    pub fn expected(&self) -> usize {
        self.expected
    }

    /// Get a reference to the raw data
    pub fn data(&self) -> &[f64] {
        &*self.input
    }
}
