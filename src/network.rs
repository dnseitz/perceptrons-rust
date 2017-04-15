
use perceptron::Perceptron;
use rand::{self, Rng};
use input::Input;
use std::ops::{Deref, DerefMut};

/*
macro_rules! neuron {
    ($name:ident[$N:expr]) => {
        struct $name {
            weights: [f64; $N],
        }
        impl $name {
            fn new() -> Self {
                let mut weights = [0f64; $N];
                for weight in &mut weights[..] {
                    *weight = rand::thread_rng().gen_range(-0.05, 0.05);
                }
                $name {
                    weights: weights,
                }
            }

            pub fn calculate(&self, input: &::input::Input) {
                assert_eq!(input.len(), self.weights.len(),
                    "Input vector passed into neuron does not have the correct length! {} weights, {} inputs",
                    self.weights.len(), input.len());

                self.weights.iter().zip(input.iter()).map(|(w, i)| w * i).sum()
            }
        }
    }
}
*/

struct Transient {
    data: Box<[f64]>,
}

impl Transient {
    fn iter(&self) -> ::std::slice::Iter<f64> {
        self.data.iter()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

struct ErrorTerm {
    error: f64,
    weights: Box<[f64]>,
}

impl ErrorTerm {
    fn new(error: f64, weights: Box<[f64]>) -> Self {
        ErrorTerm {
            error: error,
            weights: weights,
        }
    }
}

#[derive(Debug)]
enum Neuron {
    Bias,
    Hidden(Hidden),
    Output(Output),
}

impl Neuron {
    fn calculate(&self, input: &Transient) -> f64 {
        match *self {
            Neuron::Bias => 1.0,
            Neuron::Hidden(ref node) => node.calculate(input),
            Neuron::Output(ref node) => node.calculate(input),
        }
    }

    fn update(&mut self, learning_rate: f64, momentum: f64, input: &Transient, error_terms: &[(f64, f64)]) -> Option<ErrorTerm> {
        match *self {
            Neuron::Bias => None,
            Neuron::Hidden(ref mut node) => Some(node.update(learning_rate, momentum, input, error_terms)),
            Neuron::Output(ref mut node) => Some(node.update(learning_rate, momentum, input)),
        }
    }
}

#[derive(Debug)]
struct Node {
    // (prev_delta, weight)
    weights: Box<[(f64, f64)]>,
}

impl Node {
    fn new(num_weights: usize) -> Self {
        let mut weights = Vec::with_capacity(num_weights);
        for _ in 0..num_weights {
            weights.push((0.0, rand::thread_rng().gen_range(-0.05, 0.05)));
        }
        Node {
            weights: weights.into_boxed_slice(),
        }
    }

    fn fake(num_weights: usize, weight_value: f64) -> Self {
        Node { weights: vec![(0.0, weight_value); num_weights].into_boxed_slice() }
    }

    fn calculate(&self, input: &Transient) -> f64 {
        assert_eq!(input.len(), self.weights.len(),
            "Input vector passed into neuron does not have the correct length! {} weights, {} inputs",
            self.weights.len(), input.len());

        // Sigmoid activation function:
        //      sigma(w * x) = sigma(z) = 1 / (1 + e^-z)
        1.0 / ( 1.0 + ::std::f64::consts::E.powf(-self.weights.iter().zip(input.iter()).map(|(w, i)| w.1 * i).sum::<f64>()) )
    }
}

#[derive(Debug)]
struct Hidden {
    inner: Node,
}

impl Hidden {
    fn new(num_weights: usize) -> Self {
        Hidden { inner: Node::new(num_weights) }
    }

    fn fake(num_weights: usize, weight_value: f64) -> Self {
        Hidden { inner: Node::fake(num_weights, weight_value), }
    }

    fn update(&mut self, learning_rate: f64, momentum: f64, input: &Transient, error_terms: &[(f64, f64)]) -> ErrorTerm {
        let h = self.calculate(input);
        let delta_j = h*(1.0 - h)*error_terms.iter().map(|&(w, error)| w * error).sum::<f64>();
        let old_weights = self.weights.iter().map(|weight| weight.1).collect::<Vec<f64>>();
        for (weight, x) in self.weights.iter_mut().zip(input.iter()) {
            let delta = learning_rate * delta_j * x + momentum * weight.0;
            *weight = (delta, weight.1 + delta);
        }
        ErrorTerm::new(delta_j, old_weights.into_boxed_slice())
    }
}

impl Deref for Hidden {
    type Target = Node;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Hidden {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Debug)]
struct Output {
    target: f64,
    inner: Node,
}

impl Output {
    fn new(target: f64, num_weights: usize) -> Self {
        Output {
            target: target,
            inner: Node::new(num_weights),
        }
    }

    fn fake(num_weights: usize, weight_value: f64) -> Self {
        Output {
            target: 0.0,
            inner: Node::fake(num_weights, weight_value),
        }
    }

    fn update(&mut self, learning_rate: f64, momentum: f64, input: &Transient) -> ErrorTerm {
        let o = self.calculate(input);
        let delta_k = o*(1.0 - o)*(self.target - o);
        let old_weights = self.weights.iter().map(|weight| weight.1).collect::<Vec<f64>>();
        for (weight, x) in self.weights.iter_mut().zip(input.iter()) {
            let delta = learning_rate * delta_k * x + momentum * weight.0;
            *weight = (delta, weight.1 + delta);
            //*weight.1 += learning_rate * delta_k * x + momentum * prev;
        }
        ErrorTerm::new(delta_k, old_weights.into_boxed_slice())
    }
}

impl Deref for Output {
    type Target = Node;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Output {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Debug)]
struct Layer {
    nodes: Box<[Neuron]>,
}

impl Layer {
    fn new_with_bias(num_nodes: usize, input_length: usize) -> Self {
        let mut nodes = Vec::with_capacity(num_nodes);
        nodes.push(Neuron::Bias);
        for _ in 0..num_nodes {
            nodes.push(Neuron::Hidden(Hidden::new(input_length)));
        }
        Layer {
            nodes: nodes.into_boxed_slice(),
        }
    }

    fn new_output(num_nodes: usize, input_length: usize) -> Self {
        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            nodes.push(Neuron::Output(Output::new(0.5, input_length)));
        }
        Layer {
            nodes: nodes.into_boxed_slice(),
        }
    }

    fn fake_with_bias(num_nodes: usize, input_length: usize) -> Self {
        let mut nodes = Vec::with_capacity(num_nodes + 1);
        nodes.push(Neuron::Bias);
        for _ in 0..num_nodes {
            nodes.push(Neuron::Hidden(Hidden::fake(input_length, 0.1)));
        }
        Layer {
            nodes: nodes.into_boxed_slice(),
        }
    }

    fn fake_output(num_nodes: usize, input_length: usize) -> Self {
        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            nodes.push(Neuron::Output(Output::fake(input_length, 0.1)));
        }
        Layer {
            nodes: nodes.into_boxed_slice(),
        }
    }

    fn calculate(&self, input: &Transient) -> Transient {
        Transient {
            data: self.nodes.iter().map(|node| node.calculate(input)).collect::<Vec<f64>>().into_boxed_slice(),
        }
    }

    /// Update the layer, returning the error terms for this layer
    fn update(&mut self, learning_rate: f64, momentum: f64, input: &Transient, error_terms: Vec<ErrorTerm>) -> Vec<ErrorTerm> {
        self.nodes.iter_mut().enumerate().filter_map(|(i, node)| {
            let e_terms = error_terms.iter().map(|term| (term.weights[i], term.error)).collect::<Vec<(f64, f64)>>();
            node.update(learning_rate, momentum, input, &e_terms)
        }).collect::<Vec<ErrorTerm>>()
    }
}

#[derive(Debug)]
pub struct Network {
    hidden: Box<[Layer]>,
    output: Layer,
}

impl Network {
    fn new(input_len: usize, hidden: Box<[Layer]>, num_outputs: usize) -> Self {
        let num_inputs_to_output = match hidden.last() {
            Some(layer) => layer.nodes.len(),
            None => input_len,
        };
        let output = Layer::new_output(num_outputs, num_inputs_to_output);
        Network {
            hidden: hidden,
            output: output,
        }
    }

    pub fn fake() -> Self {
        let mut hidden = Vec::new();
        hidden.push(Layer::fake_with_bias(2, 3));
        let output = Layer::fake_output(1, 3);
        Network {
            hidden: hidden.into_boxed_slice(),
            output: output,
        }
    }

    pub fn fake_update(&mut self, learning_rate: f64, momentum: f64, target: f64, input: &[f64]) {
        if let Neuron::Output(ref mut node) = *self.output.nodes.iter_mut().nth(0).unwrap() {
            node.target = target;
        }
        else {
            panic!("Node in output layer wasn't an output node!");
        }
        let transient_input = Transient { data: input.iter().map(f64::clone).collect::<Vec<f64>>().into_boxed_slice() };

        update_rec(learning_rate, momentum, &transient_input, &mut self.output, &mut self.hidden[..]);
    }

    pub fn update(&mut self, learning_rate: f64, momentum: f64, input: &Input) {
        let transient_input = Transient { data: input.iter().map(f64::clone).collect::<Vec<f64>>().into_boxed_slice() };
        for (i, node) in self.output.nodes.iter_mut().enumerate() {
            if let Neuron::Output(ref mut node) = *node {
                node.target = if i == input.expected() { 0.9 } else { 0.1 }
            }
            else {
                panic!("Node in output layer wasn't an output node!");
            }
        }

        update_rec(learning_rate, momentum, &transient_input, &mut self.output, &mut self.hidden[..]);
    }

    pub fn calculate(&self, input: &Input) -> usize {
        let mut transient_input = Transient { data: input.iter().map(f64::clone).collect::<Vec<f64>>().into_boxed_slice() };

        for layer in self.hidden.iter() {
            transient_input = layer.calculate(&transient_input);
        }
        let output = self.output.calculate(&transient_input);
        output.iter()
        .enumerate()
        .max_by(|&x, &y| (x.1)
                         .partial_cmp(y.1)
                         .expect("Unable to get maximum value of output!"))
        .expect("Unable to get maximum value of output!")
        .0
    }

    pub fn calculate_accuracy(&self, data_set: &[Input]) -> f64 {
        let mut correct = 0;
        for input in data_set {
            let predicted = self.calculate(&input);
            if predicted == input.expected() {
                correct += 1;
            }
        }
        ((correct as f64) / (data_set.len() as f64))
    }
}

pub struct NetworkBuilder {
    input_len: usize,
    layers: Vec<usize>,
}

impl NetworkBuilder {
    pub fn new(input_len: usize) -> Self {
        NetworkBuilder {
            input_len: input_len,
            layers: Vec::new(),
        }
    }

    pub fn add_layer(mut self, layer_size: usize) -> Self {
        self.layers.push(layer_size);
        self
    }

    pub fn finalize(self, num_outputs: usize) -> Network {
        let mut layers: Vec<Layer> = Vec::with_capacity(self.layers.len());
        for num_nodes in self.layers {
            if layers.is_empty() {
                layers.push(Layer::new_with_bias(num_nodes, self.input_len));
            }
            else {
                let input_len = layers[layers.len() - 1].nodes.len();
                layers.push(Layer::new_with_bias(num_nodes, input_len));
            }
        }
        Network::new(self.input_len, layers.into_boxed_slice(), num_outputs)
    }
}

fn update_rec(learning_rate: f64, momentum: f64, input: &Transient, output: &mut Layer, layers: &mut [Layer]) -> Vec<ErrorTerm> {
    if layers.len() == 0 {
        //println!("Updating output layer");
        //println!("Before: {:#?}", output);
        let ret = output.update(learning_rate, momentum, input, Vec::new());
        //println!("After: {:#?}", output);
        ret
    }
    else {
        let next_input = layers[0].calculate(input);
        let error_terms = update_rec(learning_rate, momentum, &next_input, output, &mut layers[1..]);
        layers[0].update(learning_rate, momentum, input, error_terms)
    }
}

