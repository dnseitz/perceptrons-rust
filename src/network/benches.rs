
use super::*;
use test::Bencher;

#[bench]
fn bench_node_calculate(b: &mut Bencher) {
    let mut inputs = Vec::with_capacity(1000);
    for _ in 0..1000 {
        inputs.push(rand::thread_rng().gen_range(0.0, 1.0));
    }
    let data = Transient::from(inputs);
    let mut node = Node::new(1000);
    b.iter(|| node.calculate(&data));
}

#[bench]
fn bench_output_layer_calculate_25_nodes(b: &mut Bencher) {
    let mut inputs = vec![0.5; 1000];
    inputs[0] = 1.0;
    let data = Transient::from(inputs);
    let mut layer = Layer::new_output(25, 1000);
    b.iter(|| layer.calculate(&data));
}

#[bench]
fn bench_output_layer_par_calculate_25_nodes(b: &mut Bencher) {
    let mut inputs = vec![0.5; 1000];
    inputs[0] = 1.0;
    let data = Transient::from(inputs);
    let mut layer = Layer::new_output(25, 1000);
    b.iter(|| layer.par_calculate(&data));
}

#[bench]
fn bench_output_layer_update_25_nodes(b: &mut Bencher) {
    let mut inputs = vec![0.5; 1000];
    inputs[0] = 1.0;
    let data = Transient::from(inputs);
    let mut layer = Layer::new_output(25, 1000);
    b.iter(|| layer.update(0.1, 0.9, &data, Vec::new()));
}

#[bench]
fn bench_output_layer_par_update_25_nodes(b: &mut Bencher) {
    let mut inputs = vec![0.5; 1000];
    inputs[0] = 1.0;
    let data = Transient::from(inputs);
    let mut layer = Layer::new_output(25, 1000);
    b.iter(|| layer.par_update(0.1, 0.9, &data, Vec::new()));
}

#[bench]
fn bench_output_layer_calculate_50_nodes(b: &mut Bencher) {
    let mut inputs = vec![0.5; 1000];
    inputs[0] = 1.0;
    let data = Transient::from(inputs);
    let mut layer = Layer::new_output(50, 1000);
    b.iter(|| layer.calculate(&data));
}

#[bench]
fn bench_output_layer_par_calculate_50_nodes(b: &mut Bencher) {
    let mut inputs = vec![0.5; 1000];
    inputs[0] = 1.0;
    let data = Transient::from(inputs);
    let mut layer = Layer::new_output(50, 1000);
    b.iter(|| layer.par_calculate(&data));
}

#[bench]
fn bench_output_layer_calculate_100_nodes(b: &mut Bencher) {
    let mut inputs = vec![0.5; 1000];
    inputs[0] = 1.0;
    let data = Transient::from(inputs);
    let mut layer = Layer::new_output(100, 1000);
    b.iter(|| layer.calculate(&data));
}

#[bench]
fn bench_output_layer_par_calculate_100_nodes(b: &mut Bencher) {
    let mut inputs = vec![0.5; 1000];
    inputs[0] = 1.0;
    let data = Transient::from(inputs);
    let mut layer = Layer::new_output(100, 1000);
    b.iter(|| layer.par_calculate(&data));
}

#[bench]
fn bench_network_calculate(b: &mut Bencher) {
    let mut inputs = Vec::with_capacity(1000);
    for _ in 0..1000 {
        inputs.push(rand::thread_rng().gen_range(0.0, 1.0));
    }
    let data = Input::from_raw(5, &inputs);
    let mut network = NetworkBuilder::new(1000)
                                     .add_layer(100)
                                     .output(10);
    b.iter(|| network.calculate(&data));
}

#[bench]
fn bench_network_seq_calculate_accuracy(b: &mut Bencher) {
    let mut data_set = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let mut inputs = Vec::with_capacity(1000);
        for _ in 0..1000 {
            inputs.push(rand::thread_rng().gen_range(0.0, 1.0));
        }
        data_set.push(Input::from_raw(5, &inputs));
    }
    let mut network = NetworkBuilder::new(1000)
                                     .add_layer(100)
                                     .output(10);
    b.iter(|| network.calculate_accuracy(&data_set));
}

#[bench]
fn bench_network_par_calculate_accuracy(b: &mut Bencher) {
    let mut data_set = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let mut inputs = Vec::with_capacity(1000);
        for _ in 0..1000 {
            inputs.push(rand::thread_rng().gen_range(0.0, 1.0));
        }
        data_set.push(Input::from_raw(5, &inputs));
    }
    let mut network = NetworkBuilder::new(1000)
                                     .add_layer(100)
                                     .output(10);
    b.iter(|| network.par_calculate_accuracy(&data_set));
}

#[bench]
fn bench_network_update(b: &mut Bencher) {
    let mut inputs = Vec::with_capacity(1000);
    for _ in 0..1000 {
        inputs.push(rand::thread_rng().gen_range(0.0, 1.0));
    }
    let data = Input::from_raw(5, &inputs);
    let mut network = NetworkBuilder::new(1000)
                                     .add_layer(100)
                                     .output(10);
    b.iter(|| network.update(0.1, 0.9, &data));
}
