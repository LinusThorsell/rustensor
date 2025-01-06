struct Neuron {
    inputs: Vec<f64>,
    weights: Vec<f64>,
    bias: f64
}

impl Neuron {
    fn new(input_count: u32) -> Neuron {
        let mut neuron = Neuron {
            inputs: Vec::new(),
            weights: Vec::new(),
            bias: 0.0
        };

        for _ in 0..input_count {
            neuron.inputs.push(0.0);
            neuron.weights.push(rand::random::<f64>()*2.0-1.0);
        };

        neuron
    }

    fn feed_forward(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.inputs.len() {
            sum += self.inputs[i] * self.weights[i];
        }
        sum += self.bias;
        sum
    }
}

trait Layer {
    fn calculate_output(&self) -> Vec<f64>;
    fn update_inputs(&mut self, inputs: Vec<f64>) -> ();
}

struct Dense {
    neurons: Vec<Neuron>
}

impl Dense {
    fn new(input_neurons: u32, output_neurons: u32) -> Dense {
        let mut neurons = Vec::new();

        for _ in 0..output_neurons {
            neurons.push(Neuron::new(input_neurons));
        }

        Dense {
            neurons
        }
    }
}

impl Layer for Dense {
    fn calculate_output(&self) -> Vec<f64> {
        let mut output = Vec::new();
        for n in &self.neurons {
            output.push(n.feed_forward());
        }
        output
    }

    fn update_inputs(&mut self, inputs: Vec<f64>) -> () {
        for i in 0..self.neurons.len() {
            self.neurons[i].inputs = inputs.clone();
        }
    }
}

struct Network {
    layers: Vec<Dense>
}

impl Network {
    fn new(layers: Vec<Dense>) -> Network {
        Network {
            layers
        }
    }

    fn calculate_output(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.layers[0].update_inputs(input);
        let mut output = self.layers[0].calculate_output();
        for i in 1..self.layers.len() {
            self.layers[i].update_inputs(output);
            output = self.layers[i].calculate_output();
        }
        output
    }

    fn calculate_cost(&mut self, input: Vec<f64>, expected_output: Vec<f64>) -> f64 {
        let output = self.calculate_output(input);
        let mut cost = 0.0;
        for i in 0..expected_output.len() {
            cost += (output[i] - expected_output[i]).powi(2);
        }
        cost / expected_output.len() as f64
    }
}

fn main() {
    let input_layer = Dense::new(1, 5);
    let hidden_layer = Dense::new(5, 5);
    let output_layer = Dense::new(5, 1);

    let mut network = Network::new(vec![input_layer, hidden_layer, output_layer]);

    println!("{:?}", network.calculate_output(vec![1.0]));
    println!("{:?}", network.calculate_cost(vec![1.0], vec![1.0]));
}
