
use rand::{self, Rng};
use input::Input;
use std::ops::{Index, Deref, DerefMut};
use rayon::prelude::*;

pub type OutputResults = Transient;

pub struct Transient {
    data: Box<[f64]>,
}

impl Transient {
    #[cfg(test)]
    pub fn max(&self) -> f64 {
        *self.iter()
        .max_by(|x, y| x.partial_cmp(y)
                        .expect("Unable to get maximum value of output!"))
        .expect("Unable to get maximum value of output!")
    }

    pub fn max_class(&self) -> usize {
        self.iter()
        .enumerate()
        .max_by(|&x, &y| (x.1)
                         .partial_cmp(y.1)
                         .expect("Unable to get maximum value of output!"))
        .expect("Unable to get maximum value of output!")
        .0
    }

    fn iter(&self) -> ::std::slice::Iter<f64> {
        self.data.iter()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

impl Index<usize> for Transient {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a> From<&'a Input> for Transient {
    fn from(data: &Input) -> Self {
        Transient::from(data.data())
    }
}

impl From<Vec<f64>> for Transient {
    fn from(data: Vec<f64>) -> Self {
        Transient {
            data: data.into_boxed_slice(),
        }
    }
}

impl<'a> From<&'a [f64]> for Transient {
    fn from(data: &[f64]) -> Self {
        Transient {
            data: data.iter().map(f64::clone).collect::<Vec<_>>().into_boxed_slice(),
        }
    }
}

#[derive(Debug)]
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
    fn calculate(&mut self, input: &Transient) -> f64 {
        match *self {
            Neuron::Bias => 1.0,
            Neuron::Hidden(ref mut node) => node.calculate(input),
            Neuron::Output(ref mut node) => node.calculate(input),
        }
    }

    fn update(&mut self, learning_rate: f64, momentum: f64, input: &Transient, error_terms: &[(f64, f64)]) -> Option<ErrorTerm> {
        match *self {
            Neuron::Bias => None,
            Neuron::Hidden(ref mut node) => Some(node.update(learning_rate, momentum, input, error_terms)),
            Neuron::Output(ref mut node) => Some(node.update(learning_rate, momentum, input)),
        }
    }

    #[cfg(test)]
    fn get_inner_node(&self) -> Option<&Node> {
        match *self {
            Neuron::Bias => None,
            Neuron::Hidden(ref node) => Some(node),
            Neuron::Output(ref node) => Some(node),
        }
    }
}

#[derive(Debug)]
struct Node {
    // (prev_delta, weight)
    weights: Box<[(f64, f64)]>,
    output: f64,
}

impl Node {
    fn new(num_weights: usize) -> Self {
        let mut weights = Vec::with_capacity(num_weights);
        for _ in 0..num_weights {
            weights.push((0.0, rand::thread_rng().gen_range(-0.05, 0.05)));
        }
        Node {
            weights: weights.into_boxed_slice(),
            output: 0.0,
        }
    }

    fn calculate(&mut self, input: &Transient) -> f64 {
        assert_eq!(input.len(), self.weights.len(),
            "Input vector passed into neuron does not have the correct length!");

        // Sigmoid activation function:
        //      sigma(w * x) = sigma(z) = 1 / (1 + e^-z)
        self.output = 1.0 / ( 1.0 + ::std::f64::consts::E.powf(-self.weights.iter().zip(input.iter()).map(|(w, i)| w.1 * i).sum::<f64>()) );
        self.output
    }
}

impl<'a> From<&'a [f64]> for Node {
    fn from(weights: &[f64]) -> Self {
        Node {
            weights: weights.iter().map(|w| (0.0, *w)).collect::<Vec<_>>().into_boxed_slice(),
            output: 0.0,
        }
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

    fn update(&mut self, learning_rate: f64, momentum: f64, input: &Transient, error_terms: &[(f64, f64)]) -> ErrorTerm {
        let h = self.output;
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

    fn update(&mut self, learning_rate: f64, momentum: f64 , input: &Transient) -> ErrorTerm {
        let o = self.output;
        let delta_k = o*(1.0 - o)*(self.target - o);
        let old_weights = self.weights.iter().map(|weight| weight.1).collect::<Vec<f64>>();
        for (weight, x) in self.weights.iter_mut().zip(input.iter()) {
            let delta = learning_rate * delta_k * x + momentum * weight.0;
            *weight = (delta, weight.1 + delta);
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

    fn calculate(&mut self, input: &Transient) -> Transient {
        Transient::from(self.nodes.par_iter_mut().map(|node| node.calculate(input)).collect::<Vec<_>>())
    }

    /// Update the layer, returning the error terms for this layer
    fn update(&mut self, learning_rate: f64, momentum: f64, input: &Transient, error_terms: Vec<ErrorTerm>) -> Vec<ErrorTerm> {
        self.nodes.par_iter_mut().enumerate().map(|(i, node)| {
            let e_terms = error_terms.iter().map(|term| (term.weights[i], term.error)).collect::<Vec<(f64, f64)>>();
            node.update(learning_rate, momentum, input, &e_terms)
        }).filter(|e| e.is_some()).map(|e| e.unwrap()).collect::<Vec<ErrorTerm>>()
    }
}

impl From<Vec<Neuron>> for Layer {
    fn from(nodes: Vec<Neuron>) -> Self {
        Layer {
            nodes: nodes.into_boxed_slice(),
        }
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

    pub fn update(&mut self, learning_rate: f64, momentum: f64, input: &Input) {
        let transient_input = Transient::from(input);

        for (i, node) in self.output.nodes.iter_mut().enumerate() {
            if let Neuron::Output(ref mut node) = *node {
                node.target = if i == input.expected() { 0.9 } else { 0.1 }
            }
            else {
                panic!("Node in output layer wasn't an output node!");
            }
        }

        Self::update_rec(learning_rate, momentum, &transient_input, &mut self.output, &mut self.hidden[..]);
    }

    fn update_rec(learning_rate: f64, momentum: f64, input: &Transient, output: &mut Layer, layers: &mut [Layer]) -> Vec<ErrorTerm> {
        if layers.len() == 0 {
            output.calculate(input);
            output.update(learning_rate, momentum, input, Vec::new())
        }
        else {
            let next_input = layers[0].calculate(input);
            let error_terms = Self::update_rec(learning_rate, momentum, &next_input, output, &mut layers[1..]);
            layers[0].update(learning_rate, momentum, input, error_terms)
        }
    }

    pub fn calculate(&mut self, input: &Input) -> OutputResults {
        let mut transient_input = Transient::from(input);

        for layer in self.hidden.iter_mut() {
            transient_input = layer.calculate(&transient_input);
        }
        self.output.calculate(&transient_input)
    }

    pub fn calculate_accuracy(&mut self, data_set: &[Input]) -> f64 {
        let mut correct = 0;
        for input in data_set {
            let predicted = self.calculate(&input).max_class();
            if predicted == input.expected() {
                correct += 1;
            }
        }
        ((correct as f64) / (data_set.len() as f64))
    }
}

impl From<Vec<Layer>> for Network {
    fn from(mut layers: Vec<Layer>) -> Self {
        let output = layers.pop().expect("No output layer specified when trying to create network!");

        Network {
            hidden: layers.into_boxed_slice(),
            output: output,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn round_to(float: f64, digits: i32) -> f64 {
        (float * 10.0f64.powi(digits)).round() / 10.0f64.powi(digits)
    }

    #[test]
    fn test_node_calculate() {
        let data = Transient::from(&[1.0, 1.0, 0.0][..]);
        let mut node = Node::from(&[0.1, 0.1, 0.1][..]);
        assert_eq!(round_to(node.calculate(&data), 2), 0.55);
    }

    #[test]
    fn test_output_update() {
        let data = Transient::from(&[1.0, 0.55, 0.55][..]);
        let mut node = Output {
            target: 0.9,
            inner: Node::from(&[0.1, 0.1, 0.1][..]),
        };
        node.calculate(&data);
        node.update(0.2, 0.9, &data);
        assert_eq!(round_to(node.inner.weights[0].1, 4), 0.1172);
        assert_eq!(round_to(node.inner.weights[1].1, 4), 0.1095);
        assert_eq!(round_to(node.inner.weights[2].1, 4), 0.1095);
    }

    #[test]
    fn test_hidden_update() {
        let data = Transient::from(&[1.0, 1.0, 0.0][..]);
        let mut node = Hidden {
            inner: Node::from(&[0.1, 0.1, 0.1][..]),
        };
        node.calculate(&data);
        node.update(0.2, 0.9, &data, &[(0.086, 0.1)]);
        assert_eq!(round_to(node.inner.weights[0].1, 4), 0.1004);
        assert_eq!(round_to(node.inner.weights[1].1, 4), 0.1004);
        assert_eq!(round_to(node.inner.weights[2].1, 4), 0.1000);
    }

    #[test]
    fn test_layer_calculate() {
        let data = Transient::from(&[1.0, 1.0, 0.0][..]);
        let mut layer = Layer::from(vec![Neuron::Bias,
            Neuron::Hidden(Hidden { inner: Node::from(&[0.1, 0.1, 0.1][..]) }),
            Neuron::Hidden(Hidden { inner: Node::from(&[0.1, 0.1, 0.1][..]) })
        ]);
        let result = layer.calculate(&data);
        assert_eq!(round_to(result.data[0], 2), 1.00);
        assert_eq!(round_to(result.data[1], 2), 0.55);
        assert_eq!(round_to(result.data[2], 2), 0.55);
    }

    #[test]
    fn test_layer_update() {
        let data = Transient::from(&[1.0, 1.0, 0.0][..]);
        let error_terms = vec![ErrorTerm::new(0.086, vec![0.1, 0.1, 0.1].into_boxed_slice())];
        let mut layer = Layer::from(vec![Neuron::Bias,
            Neuron::Hidden(Hidden { inner: Node::from(&[0.1, 0.1, 0.1][..]) }),
            Neuron::Hidden(Hidden { inner: Node::from(&[0.1, 0.1, 0.1][..]) })
        ]);
        layer.calculate(&data);
        let new_error_terms = layer.update(0.2, 0.9, &data, error_terms);

        // First hidden node's weights
        assert_eq!(round_to(layer.nodes[1].get_inner_node().unwrap().weights[0].1, 4), 0.1004);
        assert_eq!(round_to(layer.nodes[1].get_inner_node().unwrap().weights[1].1, 4), 0.1004);
        assert_eq!(round_to(layer.nodes[1].get_inner_node().unwrap().weights[2].1, 4), 0.1000);

        // Second hidden node's weights
        assert_eq!(round_to(layer.nodes[2].get_inner_node().unwrap().weights[0].1, 4), 0.1004);
        assert_eq!(round_to(layer.nodes[2].get_inner_node().unwrap().weights[1].1, 4), 0.1004);
        assert_eq!(round_to(layer.nodes[2].get_inner_node().unwrap().weights[2].1, 4), 0.1000);

        assert_eq!(round_to(new_error_terms[0].error, 3), 0.002);
        assert_eq!(round_to(new_error_terms[1].error, 3), 0.002);
    }

    #[test]
    fn test_network_calculate() {
        let data = Input::from_raw(0, &[1.0, 1.0, 0.0]);
        let mut network = Network::from(vec![
            Layer::from(vec![Neuron::Bias,
                Neuron::Hidden(Hidden { inner: Node::from(&[0.1, 0.1, 0.1][..]) }),
                Neuron::Hidden(Hidden { inner: Node::from(&[0.1, 0.1, 0.1][..]) })
            ]),
            Layer::from(vec![Neuron::Output(Output {
                inner: Node::from(&[0.1, 0.1, 0.1][..]),
                target: 0.9,
            })])
        ]);

        let result = network.calculate(&data);

        assert_eq!(round_to(result.max(), 3), 0.552);
    }

    #[test]
    fn test_network_update() {
        let data = Input::from_raw(0, &[0.0, 1.0, 0.0]);
        let mut network = Network::from(vec![
            Layer::from(vec![Neuron::Bias,
                Neuron::Hidden(Hidden { inner: Node::from(&[0.1, 0.1, 0.1][..]) }),
                Neuron::Hidden(Hidden { inner: Node::from(&[0.1, 0.1, 0.1][..]) })
            ]),
            Layer::from(vec![Neuron::Output(Output {
                inner: Node::from(&[0.1, 0.1, 0.1][..]),
                target: 0.9,
            })])
        ]);

        network.update(0.2, 0.9, &data);

        // First hidden node's weights
        assert_eq!(round_to(network.hidden[0].nodes[1].get_inner_node().unwrap().weights[0].1, 4), 0.1004);
        assert_eq!(round_to(network.hidden[0].nodes[1].get_inner_node().unwrap().weights[1].1, 4), 0.1004);
        assert_eq!(round_to(network.hidden[0].nodes[1].get_inner_node().unwrap().weights[2].1, 4), 0.1000);

        // Second hidden node's weights
        assert_eq!(round_to(network.hidden[0].nodes[2].get_inner_node().unwrap().weights[0].1, 4), 0.1004);
        assert_eq!(round_to(network.hidden[0].nodes[2].get_inner_node().unwrap().weights[1].1, 4), 0.1004);
        assert_eq!(round_to(network.hidden[0].nodes[2].get_inner_node().unwrap().weights[2].1, 4), 0.1000);

        // Output node's weights
        assert_eq!(round_to(network.output.nodes[0].get_inner_node().unwrap().weights[0].1, 4), 0.1172);
        assert_eq!(round_to(network.output.nodes[0].get_inner_node().unwrap().weights[1].1, 4), 0.1095);
        assert_eq!(round_to(network.output.nodes[0].get_inner_node().unwrap().weights[2].1, 4), 0.1095);
    }
}
