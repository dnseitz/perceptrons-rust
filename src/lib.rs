
//! Library for building neural networks

#![feature(test)]

extern crate test;

extern crate rand;
extern crate rayon;
extern crate ndarray;
extern crate rblas;

pub const INPUT_SIZE: usize = 785;

pub mod network;
pub mod perceptron;
pub mod input;
