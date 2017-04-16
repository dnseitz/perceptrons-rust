
extern crate rand;
extern crate rayon;

//mod perceptron;
mod input;
mod network;

use std::fs::File;
use std::path::Path;
use std::io;
use std::io::prelude::*;
use input::Input;
//use perceptron::Perceptron;
use rayon::prelude::*;
use network::NetworkBuilder;

const INPUT_SIZE: usize = 785;
const NUM_EPOCHS: usize = 50;

fn read_file<P: AsRef<Path>>(path: P) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

fn write_file<P: AsRef<Path>>(path: P, data: String) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(data.as_bytes())
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

/*
fn predict(data: &Input, perceptrons: &[Perceptron]) -> usize {
    perceptrons.iter()
        .map(|perceptron| (perceptron.calculate(data), perceptron.target_class()))
        .max_by(|&(x, _), &(y, _)| x.partial_cmp(&y).expect("Unable to find max value!"))
        .expect("Unable to find max value!")
        .1
}
*/

/*
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
*/

fn exit_with_usage() -> ! {
    println!("USAGE: {} <learning_rate> [training_file test_file]", std::env::args().nth(0).unwrap());
    std::process::exit(1);
}

fn main() {
    let mut network = NetworkBuilder::new(INPUT_SIZE)
                    .add_layer(100)
                    .finalize(10);

/*
    let mut network = network::Network::fake();

    println!("{:#?}", network);
    network.fake_update(0.2, 0.9, 0.9, &[1.0, 1.0, 0.0]);
    println!("{:#?}", network);
*/

    //println!("Generated network: {:#?}", network);
    /*
    let learning_rate = match std::env::args().nth(1) {
        Some(eta) => eta.parse::<f64>().expect("learning_rate must be a number!"),
        None => exit_with_usage(),
    };
    let mut training_results: Vec<f64> = Vec::with_capacity(51);
    let mut test_results: Vec<f64> = Vec::with_capacity(51);
    */
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

    //println!("Calculating first input: {}", network.calculate(&training_inputs[0]));
    println!("Calculating Initial Training Accuracy");
    let training_accuracy = network.calculate_accuracy(&training_inputs);
    //training_results.push(training_accuracy);
    println!("Training Accuracy: {}", training_accuracy);
    let test_accuracy = network.calculate_accuracy(&test_inputs);
    println!("Calculating Initial Test Accuracy");
    //test_results.push(test_accuracy);
    println!("Test Accuracy: {}", test_accuracy);

/*
        println!("Updating Network");
        for input in training_inputs.iter() {
            network.update(0.1, 0.9, input);
        }

        return;
*/

    for _ in 0..50 {
        println!("Updating Network");
        for input in training_inputs.iter() {
            network.update(0.1, 0.9, input);
        }

        let training_accuracy = network.calculate_accuracy(&training_inputs);
        //training_results.push(training_accuracy);
        println!("Training Accuracy: {}", training_accuracy);
        let test_accuracy = network.calculate_accuracy(&test_inputs);
        //test_results.push(test_accuracy);
        println!("Test Accuracy: {}", test_accuracy);
    }

    /*
    for epoch in 1..NUM_EPOCHS + 1 {
        println!("Epoch {}:", epoch);
        for input in training_inputs.iter() {
            for perceptron in perceptrons.iter_mut() {
                perceptron.update(learning_rate, input);
            }
        }
        let training_accuracy = calculate_accuracy(&training_inputs, &perceptrons);
        training_results.push(training_accuracy);
        println!("Training Accuracy: {}", training_accuracy);
        let test_accuracy = calculate_accuracy(&test_inputs, &perceptrons);
        test_results.push(test_accuracy);
        println!("Test Accuracy: {}", test_accuracy);
    }

    let training_out = training_results.iter().map(f64::to_string).collect::<Vec<String>>().join(",");
    write_file(format!("training_eta_{}.csv", learning_rate), training_out).expect("Failed to write training data");
    let test_out = test_results.iter().map(f64::to_string).collect::<Vec<String>>().join(",");
    write_file(format!("test_eta_{}.csv", learning_rate), test_out).expect("Failed to write test data");
    */
}
