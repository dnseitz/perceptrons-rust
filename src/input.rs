
use std::fmt::{self, Debug};
use std::ops::Deref;
use INPUT_SIZE;

#[derive(Debug)]
pub struct Input {
    expected: u32,
    input: Inner,
}

struct Inner([f64; INPUT_SIZE]);

impl Input {
    /// Create a new input from a raw array
    pub fn new(raw_data: &[f64]) -> Self {
        assert!(raw_data.len() == INPUT_SIZE, "Invalid input size!");
        let expected = raw_data[0] as u32;
        let mut input = [0f64; INPUT_SIZE];
        for (i, elem) in raw_data.iter().enumerate() {
            input[i] = elem / 255f64;
        }
        input[0] = 1f64;
        Input {
            expected: expected,
            input: Inner(input),
        }
    }

    pub fn expected(&self) -> u32 {
        self.expected
    }

    pub fn iter(&self) -> ::std::slice::Iter<f64> {
        self.input.iter()
    }
}

impl Deref for Inner {
    type Target = [f64; INPUT_SIZE];

    fn deref(&self) -> &Self::Target {
        &(self.0)
    }
}

impl Debug for Inner {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let _ = write!(f, "[{}", (self.0)[0])?;
        for elem in (self.0)[1..].iter() {
            let _ = write!(f, ", {}", elem)?;
        }
        write!(f, "]")
    }
}
