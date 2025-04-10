use rustensor::activations::{ActivationFunction, Linear, ReLU};

struct Neuron {
    inputs: Vec<f64>,
    weights: Vec<f64>,
    bias: f64,
    output: f64,
    error: f64
}

impl Neuron {
    fn new(input_count: u32) -> Neuron {
        let mut neuron = Neuron {
            inputs: Vec::new(),
            weights: Vec::new(),
            bias: 0.0,
            output: 0.0,
            error: 0.0
        };

        for _ in 0..input_count {
            neuron.inputs.push(0.0);
            neuron.weights.push(rand::random::<f64>());
        };

        neuron
    }
}

trait Layer {
    fn calculate_output(&mut self) -> Vec<f64>;
    fn update_inputs(&mut self, inputs: &Vec<f64>) -> ();

    // For training network
    fn backprop_output(&mut self, expected_output: &[f64]);
    fn backprop_hidden<U: ActivationFunction>(&mut self, next_layer: Layer<Dense<U>>);
    fn update_weights(&mut self, learning_rate: f64);
}

struct Dense<T: ActivationFunction> {
    neurons: Vec<Neuron>,
    activation_function: T
}

impl<T: ActivationFunction> Dense<T> {
    fn new(input_neurons: u32, output_neurons: u32, activation_function: T) -> Self {
        let mut neurons = Vec::new();

        for _ in 0..output_neurons {
            neurons.push(Neuron::new(input_neurons));
        }

        Dense {
            neurons,
            activation_function
        }
    }
}

impl<T: ActivationFunction> Layer for Dense<T> {
    fn calculate_output(&mut self) -> Vec<f64> {
        let mut outputs = Vec::new();
        for neuron in &mut self.neurons {
            let mut sum = 0.0;
            for i in 0..neuron.inputs.len() {
                sum += neuron.inputs[i] * neuron.weights[i];
            }
            sum += neuron.bias;
            neuron.output = T::function(sum); // Apply activation
            outputs.push(neuron.output);
        }
        outputs
    }

    fn update_inputs(&mut self, inputs: &Vec<f64>) -> () {
        for i in 0..self.neurons.len() {
            self.neurons[i].inputs = inputs.clone();
        }
    }

    fn backprop_output(&mut self, expected_output: &[f64]) {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let output = neuron.output;
            let target = expected_output[i];

            let cost_gradient = output - target;

            neuron.error = cost_gradient * T::derivative(output);
        }
    }

    fn backprop_hidden<U: ActivationFunction>(&mut self, next_layer: &Layer<Dense<U>>) {
        for i in 0..self.neurons.len() {
            let mut error_sum = 0.0;

            for neuron_next in &next_layer.neurons {
                error_sum += neuron_next.weights[i] * neuron_next.error;
            }

            self.neurons[i].error = error_sum * T::derivative(self.neurons[i].output);
        }
    }

    fn update_weights(&mut self, learning_rate: f64) {
        for n in &mut self.neurons {
            for i in 0..n.weights.len() {
                n.weights[i] -= learning_rate * n.error * n.inputs[i];
            }
            n.bias -= learning_rate * n.error;
        }
    }
}

// struct Network {
//     layers: Vec<Dense<dyn ActivationFunction>>
// }
//
// impl Network {
//     fn new(layers: Vec<Dense>) -> Network {
//         Network {
//             layers
//         }
//     }
//
//     fn calculate_output(&mut self, input: &Vec<f64>) -> Vec<f64> {
//         // Input layer
//         self.layers[0].update_inputs(input);
//         let mut output = self.layers[0].calculate_output(ReLU::function);
//
//         // Hidden layers
//         for i in 1..self.layers.len()-1 {
//             self.layers[i].update_inputs(&output);
//             output = self.layers[i].calculate_output(ReLU::function);
//         }
//
//         // Output layer - No relu for last layer
//         let last_layer_index = self.layers.len()-1;
//         self.layers[last_layer_index].update_inputs(&output);
//         output = self.layers[last_layer_index].calculate_output(Linear::function);
//
//         output
//     }
//
//     /*
//     * Mean squared error
//     */
//     fn calculate_cost(&mut self, input: &Vec<f64>, expected_output: &Vec<f64>) -> f64 {
//         let output = self.calculate_output(input);
//         let mut cost = 0.0;
//         for i in 0..expected_output.len() {
//             cost += (output[i] - expected_output[i]).powi(2);
//         }
//         cost / expected_output.len() as f64
//     }
//
//     fn train(&mut self, input: &Vec<f64>, expected_output: &Vec<f64>, learning_rate: f64) -> f64 {
//         let cost = self.calculate_cost(input, expected_output);
//
//         let last_layer_index = self.layers.len() - 1;
//         self.layers[last_layer_index].backprop_output(&expected_output);
//
//         for i in (0..last_layer_index).rev() {
//             let (left, right) = self.layers.split_at_mut(i + 1);
//             let current_layer = &mut left[i];       // layer i
//             let next_layer = &right[0];             // layer i+1
//             current_layer.backprop_hidden(next_layer);
//         }
//
//         for layer in &mut self.layers {
//             layer.update_weights(learning_rate);
//         }
//
//         cost
//     }
// }
//
fn main() {
    let input_layer = Dense::new(1, 2, ReLU);
    let hidden_layer_1 = Dense::new(2, 2, ReLU);
    let hidden_layer_2 = Dense::new(2, 2, ReLU);
    let output_layer = Dense::new(2, 1, ReLU);

    // let mut network = Network::new(vec![
    //     input_layer,
    //     hidden_layer_1,
    //     hidden_layer_2,
    //     output_layer
    // ]);

    let learning_rate = 0.001;
    let epochs = 250;

    let inputs = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0], vec![6.0], vec![7.0], vec![8.0], vec![9.0]];
    let outputs = vec![vec![3.0], vec![7.0], vec![11.0], vec![15.0], vec![19.0], vec![23.0], vec![27.0], vec![31.0], vec![35.0], vec![39.0]];

    // for _ in 0..epochs {
    //     for i in 0..inputs.len() {
    //         network.train(&inputs[i], &outputs[i], learning_rate);
    //     }
    //     println!("Input: {:?}", &inputs[0]);
    //     println!("Output: {:?}", network.calculate_output(&inputs[0]));
    //     println!("Weights: {:?}", network.layers[1].neurons[0].weights);
    //     println!("Bias: {:?}", network.layers[1].neurons[0].bias);
    //     println!("Cost: {:?}", network.calculate_cost(&inputs[0], &outputs[0]));
    // }
}
