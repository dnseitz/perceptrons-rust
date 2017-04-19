
#[cfg(test)]
mod tests;
#[cfg(test)]
mod benches;

use rand::{self, Rng};
use input::Input;
use std::ops::{Index, Deref, DerefMut};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
//use ndarray::prelude::*;

const BIAS_VALUE: f64 = 1.0;

const RANDOM_RANGE_HIGH: f64 = 0.05;
const RANDOM_RANGE_LOW:  f64 = -0.05;

const TARGET_HIGH: f64 = 0.9;
const TARGET_LOW:  f64 = 0.1;

/// Output result from running some input through the whole network.
pub type OutputResults = Transient;

impl OutputResults {
    /// Get the maximum value out of the vector of data.
    #[cfg(test)]
    pub fn max(&self) -> f64 {
        *self.iter()
             .max_by(|x, y| x.partial_cmp(y)
                             .expect("Unable to get maximum value of output!"))
             .expect("Unable to get maximum value of output!")
    }

    /// Get the index of the maximum value in the vector of data
    pub fn max_class(&self) -> usize {
        self.iter()
        .enumerate()
        .max_by(|&x, &y| (x.1)
                         .partial_cmp(y.1)
                         .expect("Unable to get maximum value of output!"))
        .expect("Unable to get maximum value of output!")
        .0
    }
}

/// A struct packing an array of data to be passed into a layer of the network.
pub struct Transient {
    data: Box<[f64]>,
    //data: Array1<f64>,
}

impl Transient {
    fn iter(&self) -> /*::ndarray::iter::Iter<f64, ::ndarray::Dim<[usize; 1]>> {*/ ::std::slice::Iter<f64> {
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
            //data: Array::from_vec(data),
        }
    }
}

impl<'a> From<&'a [f64]> for Transient {
    fn from(data: &[f64]) -> Self {
        Transient {
            data: data.iter().map(f64::clone).collect::<Vec<_>>().into_boxed_slice(),
            //data: Array::from_iter(data.iter().map(f64::clone)),
        }
    }
}

/// A struct packing the error term of a node along with all its previous weights
///
/// Used in back propagation to calculate the updated weights of hidden nodes.
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

/// An enum classifying the different types of neurons
#[derive(Clone, Debug)]
enum Neuron {
    Bias,
    Hidden(HiddenNode),
    Output(OutputNode),
}

impl Neuron {
    fn cached_calculate(&mut self, input: &Transient) -> f64 {
        match *self {
            Neuron::Bias => BIAS_VALUE,
            Neuron::Hidden(ref mut node) => node.cached_calculate(input),
            Neuron::Output(ref mut node) => node.cached_calculate(input),
        }
    }

    fn calculate(&self, input: &Transient) -> f64 {
        match *self {
            Neuron::Bias => BIAS_VALUE,
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

    #[cfg(test)]
    fn get_inner_node(&self) -> Option<&Node> {
        match *self {
            Neuron::Bias => None,
            Neuron::Hidden(ref node) => Some(node),
            Neuron::Output(ref node) => Some(node),
        }
    }
}

/// A basic node that can calulate output based on some input values
#[derive(Clone, Debug)]
struct Node {
    // (prev_delta, weight)
    weights: Box<[(f64, f64)]>,
    //weights: Array1<(f64, f64)>,
    output: f64,
}

impl Node {
    // Create a new node with the given number of weights
    fn new(num_weights: usize) -> Self {
        let mut weights = Vec::with_capacity(num_weights);
        for _ in 0..num_weights {
            weights.push((0.0, rand::thread_rng().gen_range(RANDOM_RANGE_LOW, RANDOM_RANGE_HIGH)));
        }
        Node {
            weights: weights.into_boxed_slice(),
            //weights: Array::from_vec(weights),
            output: 0.0,
        }
    }

    // Calculate the output for this node with a given input vector
    fn cached_calculate(&mut self, input: &Transient) -> f64 {
        debug_assert_eq!(input.len(), self.weights.len(),
            "Input vector passed into neuron does not have the correct length!");
        //let z = self.weights.iter().map(|&(_, w)| w).collect::<Array1<f64>>().dot(&input.data);

        // Sigmoid activation function:
        //      sigma(w * x) = sigma(z) = 1 / (1 + e^-z)
        //
        // Optimization: cache the output for use in the update function
        self.output = self.calculate(input);
        //self.output = 1.0 / ( 1.0 + ::std::f64::consts::E.powf(-z) );
        self.output
    }

    fn calculate(&self, input: &Transient) -> f64 {
        debug_assert_eq!(input.len(), self.weights.len(),
            "Input vector passed into neuron does not have the correct length!");

        1.0 / ( 1.0 + ::std::f64::consts::E.powf(-self.weights.iter().zip(input.iter()).map(|(w, i)| w.1 * i).sum::<f64>()) )
    }
}

impl<'a> From<&'a [f64]> for Node {
    fn from(weights: &[f64]) -> Self {
        Node {
            weights: weights.iter().map(|&w| (0.0, w)).collect::<Vec<_>>().into_boxed_slice(),
            //weights: Array::from_iter(weights.iter().map(|&w| (0.0, w))),
            output: 0.0,
        }
    }
}

/// A hidden node
#[derive(Clone, Debug)]
struct HiddenNode {
    inner: Node,
}

impl HiddenNode {
    // Create a new hidden node with the given number of weights
    fn new(num_weights: usize) -> Self {
        HiddenNode { inner: Node::new(num_weights) }
    }

    // Update the neuron with the given input data, `calculate` MUST be called before calling this
    // method.
    fn update(&mut self,
        learning_rate: f64,
        momentum: f64,
        input: &Transient,
        error_terms: &[(f64, f64)])
        -> ErrorTerm
    {
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

impl Deref for HiddenNode {
    type Target = Node;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for HiddenNode {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// An output node
#[derive(Clone, Debug)]
struct OutputNode {
    target: f64,
    inner: Node,
}

impl OutputNode {
    // Create a new output node with the given target value and weights
    fn new(target: f64, num_weights: usize) -> Self {
        OutputNode {
            target: target,
            inner: Node::new(num_weights),
        }
    }

    // Update the neuron with the given input data, `calculate` MUST be called before calling this
    // method
    fn update(&mut self, learning_rate: f64, momentum: f64 , input: &Transient) -> ErrorTerm {
        let o = self.output;
        let error_term = o*(1.0 - o)*(self.target - o);
        let old_weights = self.weights.iter().map(|weight| weight.1).collect::<Vec<f64>>();

        // Update all the weights
        for (weight, x) in self.weights.iter_mut().zip(input.iter()) {
            let delta = learning_rate * error_term * x + momentum * weight.0;
            *weight = (delta, weight.1 + delta);
        }
        ErrorTerm::new(error_term, old_weights.into_boxed_slice())
    }
}

impl Deref for OutputNode {
    type Target = Node;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for OutputNode {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// A layer encapsulating an array of `Neuron`s
#[derive(Clone, Debug)]
struct Layer {
    nodes: Box<[Neuron]>,
}

impl Layer {
    // Create a new layer with a bias node
    //
    // Each node takes `input_length` inputs.
    fn new_with_bias(num_nodes: usize, input_length: usize) -> Self {
        let mut nodes = Vec::with_capacity(num_nodes);
        nodes.push(Neuron::Bias);
        for _ in 0..num_nodes {
            nodes.push(Neuron::Hidden(HiddenNode::new(input_length)));
        }
        Layer {
            nodes: nodes.into_boxed_slice(),
        }
    }

    // Create a new output layer (no bias node)
    //
    // Each node takes `input_length` inputs.
    fn new_output(num_nodes: usize, input_length: usize) -> Self {
        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            nodes.push(Neuron::Output(OutputNode::new(0.5, input_length)));
        }
        Layer {
            nodes: nodes.into_boxed_slice(),
        }
    }

    fn cached_calculate(&mut self, input: &Transient) -> Transient {
        Transient::from(
            self.nodes.iter_mut()
                      .map(|node| node.cached_calculate(input))
                      .collect::<Vec<_>>()
        )
    }

    // Calculate the output values for each node in this layer
    fn calculate(&self, input: &Transient) -> Transient {
        Transient::from(
            self.nodes.iter()
                       .map(|node| node.calculate(input))
                       .collect::<Vec<_>>()
        )
    }

    fn par_cached_calculate(&mut self, input: &Transient) -> Transient {
        Transient::from(
            self.nodes.par_iter_mut()
                      .map(|node| node.cached_calculate(input))
                      .collect::<Vec<_>>()
        )
    }

    // Parallel version of calculate
    fn par_calculate(&self, input: &Transient) -> Transient {
        Transient::from(
            self.nodes.par_iter()
                       .map(|node| node.calculate(input))
                       .collect::<Vec<_>>()
        )
    }

    // Update the layer, returning the error terms for this layer
    fn update(&mut self, learning_rate: f64, momentum: f64, input: &Transient, error_terms: Vec<ErrorTerm>) -> Vec<ErrorTerm> {
        self.nodes.iter_mut()
                  .enumerate()
                  .map(|(i, node)| {
                      let e_terms = error_terms.iter()
                                               .map(|term| (term.weights[i], term.error))
                                               .collect::<Vec<(f64, f64)>>();
                      node.update(learning_rate, momentum, input, &e_terms)
                  })
                  .filter(|e| e.is_some())
                  .map(|e| e.unwrap())
                  .collect::<Vec<ErrorTerm>>()
    }

    fn par_update(&mut self, learning_rate: f64, momentum: f64, input: &Transient, error_terms: Vec<ErrorTerm>) -> Vec<ErrorTerm> {
        self.nodes.par_iter_mut()
                  .enumerate()
                  .map(|(i, node)| {
                      let e_terms = error_terms.iter()
                                               .map(|term| (term.weights[i], term.error))
                                               .collect::<Vec<(f64, f64)>>();
                      node.update(learning_rate, momentum, input, &e_terms)
                  })
                  .filter(|e| e.is_some())
                  .map(|e| e.unwrap())
                  .collect::<Vec<ErrorTerm>>()
    }
}

impl From<Vec<Neuron>> for Layer {
    fn from(nodes: Vec<Neuron>) -> Self {
        Layer {
            nodes: nodes.into_boxed_slice(),
        }
    }
}

/// A neural network able to represent several hidden layers as well as an output layer.
#[derive(Clone, Debug)]
pub struct Network {
    hidden: Box<[Layer]>,
    output: Layer,
}

impl Network {
    // Create a new neural network with the given input layer length, hidden layers and output
    // layer size
    fn new(input_len: usize, hidden: Box<[Layer]>, num_outputs: usize) -> Self {
        // The number of inputs to our output layer is either the size of our last hidden layer, or
        // our input layer size if we have no hidden layers.
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

    /// Update the weights in the network.
    ///
    /// Recalculates weights for each node in the network with the given `learning_rate` and
    /// `momentum` values based on the `input`.
    pub fn update(&mut self, learning_rate: f64, momentum: f64, input: &Input) {
        let transient_input = Transient::from(input);

        for (i, node) in self.output.nodes.iter_mut().enumerate() {
            if let Neuron::Output(ref mut node) = *node {
                node.target = if i == input.expected() { TARGET_HIGH } else { TARGET_LOW }
            }
            else {
                panic!("Node in output layer wasn't an output node!");
            }
        }

        Self::update_rec(learning_rate,
            momentum,
            &transient_input,
            &mut self.output,
            &mut self.hidden[..]);
    }

    // Recursive implementation of update, we forward propagate the input as we travel through
    // each layer of the network, then use back propagation to update the weights of each node.
    fn update_rec(learning_rate: f64,
        momentum: f64,
        input: &Transient,
        output: &mut Layer,
        layers: &mut [Layer])
        -> Vec<ErrorTerm>
    {
        const PAR_CALC_THRESHOLD: usize = 25;
        const PAR_THRESHOLD: usize = 75;
        // Base case, no hidden layers left, update output layer
        if layers.len() == 0 {
            // We need to calculate the output value in order to cache it for each node
            output.cached_calculate(input);
            output.update(learning_rate, momentum, input, Vec::new())
        }
        else {
            let next_input = if layers[0].nodes.len() > PAR_CALC_THRESHOLD {
                layers[0].par_cached_calculate(input)
            }
            else {
                layers[0].cached_calculate(input)
            };

            let error_terms = Self::update_rec(learning_rate,
                                  momentum,
                                  &next_input,
                                  output,
                                  &mut layers[1..]);

            // Back propagation step
            if layers[0].nodes.len() > PAR_THRESHOLD {
                layers[0].par_update(learning_rate, momentum, input, error_terms)
            }
            else {
                layers[0].update(learning_rate, momentum, input, error_terms)
            }
        }
    }

    pub fn calculate(&self, input: &Input) -> OutputResults {
        const PAR_CALC_THRESHOLD: usize = 25;
        const PAR_THRESHOLD: usize = 75;

        let mut transient_input = Transient::from(input);

        for layer in self.hidden.iter() {
            transient_input = if layer.nodes.len() > PAR_CALC_THRESHOLD {
                layer.par_calculate(&transient_input)
            }
            else {
                layer.calculate(&transient_input)
            };
        }
        if self.output.nodes.len() > PAR_CALC_THRESHOLD {
            self.output.par_calculate(&transient_input)
        }
        else {
            self.output.calculate(&transient_input)
        }
    }

    pub fn calculate_accuracy(&self, data_set: &[Input]) -> f64 {
        const PAR_SET_SIZE_THRESHOLD: usize = 1000;
        if data_set.len() > PAR_SET_SIZE_THRESHOLD {
            self.par_calculate_accuracy(data_set)
        }
        else {
            self.seq_calculate_accuracy(data_set)
        }
    }

    fn seq_calculate_accuracy(&self, data_set: &[Input]) -> f64 {
        let mut correct = 0;
        for input in data_set {
            let predicted = self.calculate(&input).max_class();
            if predicted == input.expected() {
                correct += 1;
            }
        }
        ((correct as f64) / (data_set.len() as f64))
    }

    fn par_calculate_accuracy(&self, data_set: &[Input]) -> f64 {
        let correct: AtomicUsize = AtomicUsize::new(0);
        data_set.par_iter().for_each(|input| {
            let predicted = self.calculate(&input).max_class();
            if predicted == input.expected() {
                correct.fetch_add(1, Ordering::Relaxed);
            }
        });
        ((correct.load(Ordering::Relaxed) as f64) / (data_set.len() as f64))
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

/// Use to create a new network.
///
/// Layers are specified from the input to the output, so the first layer specified will take its
/// inputs from the input layer, the second will take its inputs from the first hidden layer and so
/// on until the output layer, which takes its inputs from the last hidden layer (or the input
/// layer if there are no hidden layers).
pub struct NetworkBuilder {
    input_len: usize,
    layers: Vec<usize>,
}

impl NetworkBuilder {
    /// Create a new Network with a given input layer size.
    pub fn new(input_len: usize) -> Self {
        NetworkBuilder {
            input_len: input_len,
            layers: Vec::new(),
        }
    }

    /// Add a new input layer with the given size.
    pub fn add_layer(mut self, layer_size: usize) -> Self {
        self.layers.push(layer_size);
        self
    }

    /// Add the output layer with the given layer size.
    ///
    /// Return the `Network` with randomly generated weights.
    pub fn output(self, num_outputs: usize) -> Network {
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
