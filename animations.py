from manim import *



class NetworkVisualization(Scene):
    def construct(self):
        from main import neural_net

        # Convert the network directly into a Manim graph
        self.create_network_graph(neural_net)

        # Wait before ending the animation
        self.wait(2)

    def create_network_graph(self, network):
        """
        Converts a neural network object into a Manim graph.
        This avoids using NetworkX and directly creates the graph using Dot and Line objects.
        """
        layers = {}  # To store nodes for each layer

        # Set up spacing for the network
        x_spacing = 2  # Horizontal spacing between layers
        y_spacing = 1  # Vertical spacing between neurons in the same layer

        # Create input layer nodes (since these are not in the 'layers' list of the network object)
        input_layer_nodes = []
        for i in range(network.n_inputs):
            input_node = Dot(radius=0.2, color=GREEN)  # Input neurons in green
            input_layer_nodes.append(input_node)
            self.add(input_node)  # Add to the scene

        layers[0] = input_layer_nodes  # Input layer is at index 0

        # Create nodes for subsequent layers
        for i, layer in enumerate(network.layers):
            layer_nodes = []
            for j, neuron in enumerate(layer.neurons):
                node = Dot(radius=0.2, color=BLUE)  # Neurons in blue
                layer_nodes.append(node)
                self.add(node)  # Add to the scene

            layers[i + 1] = layer_nodes  # Start from index 1 for hidden layers, 2 for the output layer, etc.

        # Position the nodes horizontally (left to right)
        for i, layer_nodes in layers.items():
            # Calculate the x position (based on layer index) and y position (based on neuron index)
            x_pos = (i-(len(layers)-1)/2) * x_spacing  # Layer spacing
            y_pos = (len(layer_nodes) / 2) * y_spacing  # Centering the nodes vertically

            for j, node in enumerate(layer_nodes):
                # Adjust the y position of each neuron to space them out evenly
                node.move_to(UP * (j - len(layer_nodes) / 2) * y_spacing + RIGHT * x_pos)

        # Position the input layer nodes at x = 0
        for i, node in enumerate(input_layer_nodes):
            y_pos = (i - len(input_layer_nodes) / 2) * y_spacing
            node.move_to(UP * y_pos + RIGHT * ((1-len(layers))/2 * (x_spacing)))  # Position at x = 0

        # Add edges between neurons in adjacent layers
        for layer_idx in range(len(layers) - 1):  # Exclude the last layer, since it has no outgoing edges
            current_layer = layers[layer_idx]
            next_layer = layers[layer_idx + 1]

            for current_neuron in current_layer:
                for next_neuron in next_layer:
                    edge = Line(current_neuron.get_center(), next_neuron.get_center(), color=WHITE)
                    self.add(edge)