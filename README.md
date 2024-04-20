# Virtual Logic Circuit with Falcon Language Model

This project demonstrates the construction of a virtual logic circuit using the Falcon language model from Anthropic. The Falcon model is used to dynamically generate the logic for composing and simulating the logic circuit based on user-provided inputs.

## Features

- **Logic Gate Models**: The project defines separate models for common logic gates, such as AND, OR, and NOT gates.
- **Circuit Composition Model**: This model is responsible for combining the individual logic gate models into a larger circuit.
- **Circuit Simulation Model**: This model takes the composed circuit and simulates its behavior, providing outputs for given inputs.
- **Language Model-Driven Logic Generation**: The Falcon language model is used to generate the necessary steps for composing and simulating the logic circuit, allowing for dynamic and adaptable circuit design.

## Usage

1. Ensure you have the necessary dependencies installed, including PyTorch and the Langchain library.
2. Set up your HuggingFaceHub API token as an environment variable (`HUGGINGFACEHUB_API_TOKEN`).
3. Run the `virtual_logic.py` script to see an example of the virtual logic circuit in action.

```python
import torch
from virtual_logic import CircuitSimulator, HuggingFaceHub

llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)
circuit_sim = CircuitSimulator(llm)

input1 = torch.tensor([0.0, 1.0])
input2 = torch.tensor([1.0, 0.0])
output = circuit_sim(input1, input2)
print(output)
```

## Customization

You can customize the virtual logic circuit by:

1. Adding or modifying the existing logic gate models.
2. Enhancing the `CircuitComposer` class to support more complex circuit topologies.
3. Updating the prompting and logic generation in the `CircuitSimulator` class to explore different approaches for leveraging the language model.
4. Integrating the virtual logic circuit into larger applications or systems that require dynamic and adaptable logic processing.

## Contributing

Contributions to this project are welcome. If you have any ideas, suggestions, or improvements, please feel free to create a new issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).