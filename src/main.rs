
extern crate perceptron;

/*
mod input;
mod network;
*/
extern crate rayon;

use std::fs::File;
use std::path::Path;
use std::io;
use std::io::prelude::*;
use perceptron::input::Input;
use rayon::prelude::*;
use perceptron::network::NetworkBuilder;
use perceptron::INPUT_SIZE;

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

fn exit_with_usage() -> ! {
    println!("USAGE: {} <num_hidden_nodes> [training_file test_file]", std::env::args().nth(0).unwrap());
    std::process::exit(1);
}

fn main() {
/*
    let learning_rate = match std::env::args().nth(1) {
        Some(eta) => eta.parse::<f64>().expect("learning_rate must be a number!"),
        None => exit_with_usage(),
    };
*/
    let num_hidden_nodes = match std::env::args().nth(1) {
        Some(n) => n.parse::<usize>().expect("number of hidden nodes must be a number!"),
        None => exit_with_usage(),
    };

    let mut network = NetworkBuilder::input_layer(INPUT_SIZE)
                                     .hidden_layer(num_hidden_nodes)
                                     .output_layer(10);
    let mut training_results: Vec<f64> = Vec::with_capacity(51);
    let mut test_results: Vec<f64> = Vec::with_capacity(51);

    let training_filename = "mnist_train.csv";
    let test_filename = "mnist_test.csv";
    println!("Reading from {}", training_filename);
    let training_data = read_file(training_filename).expect("Failed to read training file");
    println!("Reading from {}", test_filename);
    let test_data = read_file(test_filename).expect("Failed to read test file");

    println!("Parsing training data");
    let training_inputs: Vec<Input> = parse_csv(training_data).par_iter().map(|row| Input::from_greyscale(&row)).collect();
    println!("Parsing test data");
    let test_inputs: Vec<Input> = parse_csv(test_data).iter().map(|row| Input::from_greyscale(&row)).collect();

    println!("Calculating Initial Training Accuracy");
    let training_accuracy = network.calculate_accuracy(&training_inputs);
    training_results.push(training_accuracy);
    println!("Training Accuracy: {}", training_accuracy);
    let test_accuracy = network.calculate_accuracy(&test_inputs);
    println!("Calculating Initial Test Accuracy");
    test_results.push(test_accuracy);
    println!("Test Accuracy: {}", test_accuracy);

    for epoch in 1..NUM_EPOCHS+1 {
        println!("Running Epoch: {}", epoch);
        println!("Updating Network");
        for input in training_inputs.iter() {
            network.update(0.1, 0.9, input);
        }

        let training_accuracy = network.calculate_accuracy(&training_inputs);
        training_results.push(training_accuracy);
        println!("Training Accuracy: {}", training_accuracy);
        let test_accuracy = network.calculate_accuracy(&test_inputs);
        test_results.push(test_accuracy);
        println!("Test Accuracy: {}", test_accuracy);
    }

    /*
    let training_out = training_results.iter().map(f64::to_string).collect::<Vec<String>>().join(",");
    write_file(format!("training_eta_{}.csv", learning_rate), training_out).expect("Failed to write training data");
    let test_out = test_results.iter().map(f64::to_string).collect::<Vec<String>>().join(",");
    write_file(format!("test_eta_{}.csv", learning_rate), test_out).expect("Failed to write test data");
    */
}
