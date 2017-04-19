
use super::*;

fn round_to(float: f64, digits: i32) -> f64 {
    (float * 10.0f64.powi(digits)).round() / 10.0f64.powi(digits)
}

#[test]
fn test_node_calculate() {
    let data = Transient::from(&[1.0, 1.0, 0.0][..]);
    let node = Node::from(&[0.1, 0.1, 0.1][..]);
    assert_eq!(round_to(node.calculate(&data), 2), 0.55);
}

#[test]
fn test_output_update() {
    let data = Transient::from(&[1.0, 0.55, 0.55][..]);
    let mut node = OutputNode {
        target: 0.9,
        inner: Node::from(&[0.1, 0.1, 0.1][..]),
    };
    node.cached_calculate(&data);
    node.update(0.2, 0.9, &data);
    assert_eq!(round_to(node.inner.weights[0].1, 4), 0.1172);
    assert_eq!(round_to(node.inner.weights[1].1, 4), 0.1095);
    assert_eq!(round_to(node.inner.weights[2].1, 4), 0.1095);
}

#[test]
fn test_hidden_update() {
    let data = Transient::from(&[1.0, 1.0, 0.0][..]);
    let mut node = HiddenNode {
        inner: Node::from(&[0.1, 0.1, 0.1][..]),
    };
    node.cached_calculate(&data);
    node.update(0.2, 0.9, &data, &[(0.086, 0.1)]);
    assert_eq!(round_to(node.inner.weights[0].1, 4), 0.1004);
    assert_eq!(round_to(node.inner.weights[1].1, 4), 0.1004);
    assert_eq!(round_to(node.inner.weights[2].1, 4), 0.1000);
}

#[test]
fn test_layer_calculate() {
    let data = Transient::from(&[1.0, 1.0, 0.0][..]);
    let layer = Layer::from(vec![Neuron::Bias,
        Neuron::Hidden(HiddenNode { inner: Node::from(&[0.1, 0.1, 0.1][..]) }),
        Neuron::Hidden(HiddenNode { inner: Node::from(&[0.1, 0.1, 0.1][..]) })
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
        Neuron::Hidden(HiddenNode { inner: Node::from(&[0.1, 0.1, 0.1][..]) }),
        Neuron::Hidden(HiddenNode { inner: Node::from(&[0.1, 0.1, 0.1][..]) })
    ]);
    layer.cached_calculate(&data);
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
    let network = Network::from(vec![
        Layer::from(vec![Neuron::Bias,
            Neuron::Hidden(HiddenNode { inner: Node::from(&[0.1, 0.1, 0.1][..]) }),
            Neuron::Hidden(HiddenNode { inner: Node::from(&[0.1, 0.1, 0.1][..]) })
        ]),
        Layer::from(vec![Neuron::Output(OutputNode {
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
            Neuron::Hidden(HiddenNode { inner: Node::from(&[0.1, 0.1, 0.1][..]) }),
            Neuron::Hidden(HiddenNode { inner: Node::from(&[0.1, 0.1, 0.1][..]) })
        ]),
        Layer::from(vec![Neuron::Output(OutputNode {
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
