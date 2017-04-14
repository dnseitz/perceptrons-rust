
extern crate rand;
extern crate rayon;

mod perceptron;
mod input;

use std::fs::File;
use std::path::Path;
use std::io;
use std::io::prelude::*;
use input::Input;
use perceptron::Perceptron;
use rayon::prelude::*;

const INPUT_SIZE: usize = 785;
const NUM_EPOCHS: usize = 50;

fn read_file<P: AsRef<Path>>(path: P) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

/// Parse a csv formatted file passed in as a string
///
/// The format of `contents` should be lines of comma separated f64 values.
fn parse_csv(contents: String) -> Vec<Vec<f64>> {
    contents.split("\n")
    .filter(|row| !row.is_empty())
    .map(|row| row.split(",")
               .map(|elem| elem.parse::<f64>()
                           .expect("Failed to parse input data"))
                           .collect())
    .collect()
}

fn predict(data: &Input, perceptrons: &[Perceptron]) -> u32 {
    perceptrons.iter()
        .map(|perceptron| (perceptron.calculate(data), perceptron.target_class()))
        .max_by(|&(x, _), &(y, _)| x.partial_cmp(&y).expect("Unable to find max value!"))
        .expect("Unable to find max value!")
        .1
}

fn calculate_accuracy(data_set: &[Input], perceptrons: &[Perceptron]) -> f64 {
    let mut correct = 0;
    for input in data_set {
        let predicted = predict(input, perceptrons);
        if predicted == input.expected() {
            correct += 1;
        }
    }
    ((correct as f64) / (data_set.len() as f64))
}

fn main() {
    let mut perceptrons = [
        Perceptron::new(0),
        Perceptron::new(1),
        Perceptron::new(2),
        Perceptron::new(3),
        Perceptron::new(4),
        Perceptron::new(5),
        Perceptron::new(6),
        Perceptron::new(7),
        Perceptron::new(8),
        Perceptron::new(9),
    ];
    let training_filename = "mnist_train.csv";
    let test_filename = "mnist_test.csv";
    println!("Reading from {}", training_filename);
    let training_data = read_file(training_filename).expect("Failed to read training file");
    println!("Reading from {}", test_filename);
    let test_data = read_file(test_filename).expect("Failed to read test file");

    println!("Parsing training data");
    let training_inputs: Vec<Input> = parse_csv(training_data).par_iter().map(|row| Input::new(&row)).collect();
    println!("Parsing test data");
    let test_inputs: Vec<Input> = parse_csv(test_data).iter().map(|row| Input::new(&row)).collect();

    println!("Calculating Initial Accuracy");
    println!("Training Accuracy: {}", calculate_accuracy(&training_inputs, &perceptrons));
    println!("Test Accuracy: {}", calculate_accuracy(&test_inputs, &perceptrons));

    for epoch in 1..NUM_EPOCHS + 1 {
        println!("Epoch {}:", epoch);
        for input in training_inputs.iter() {
            for perceptron in perceptrons.iter_mut() {
                perceptron.update(0.01, input);
            }
        }
        println!("Training Accuracy: {}", calculate_accuracy(&training_inputs, &perceptrons));
        println!("Test Accuracy: {}", calculate_accuracy(&test_inputs, &perceptrons));
    }
}
