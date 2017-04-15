
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
    /// Create a new input from a raw array
    pub fn new(raw_data: &[f64]) -> Self {
        let expected = raw_data[0] as usize;
        let mut input = raw_data.iter().map(|elem| elem / 255.0).collect::<Vec<f64>>().into_boxed_slice();
        input[0] = 1.0;
        Input {
            expected: expected,
            input: input,
        }
    }

    pub fn expected(&self) -> usize {
        self.expected
    }

    pub fn iter(&self) -> ::std::slice::Iter<f64> {
        self.input.iter()
    }

    pub fn len(&self) -> usize {
        self.input.len()
    }
}

/*
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
*/
