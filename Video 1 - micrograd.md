# Video 1 - Micrograd

https://youtu.be/VMj-3S1tku0

Micrograd allows you to compute the partial derivative of any expression (which we can think of as a graph). A neural network is nothing but an expression DAG. Micrograd operates only on scalar values, so a neural network has to be broken down into its simplest mathematical operations.

> Micrograd is all you need to train neural networks. Everything else is just efficiency.

### Understanding a derivative

Let’s define a simple scalar value function:

$$f(x) = 3x^2 - 4x + 5$$

We’re not going to bother with the analytical/symbolic derivative of this. “No one in neural networks actually writes out the expression – it would be too massive.”

The numerical derivative measures how much the function $f(x)$ responds to a small change in the input variable $x$. 

```julia
function dx(f, x)
    h = 1e-6
    return (f(x + h) - f(x)) / h
end
```

Now let’s consider an expression with three scalar input variables, not just one: `d = a * b + c`

We can compute derivatives of this in the same way, but with respect to one input variable at a time.

### Value and visualization

We define a value object that has a `label` and a value (`data`). We also define the operators `+` and `*` on value objects. The value objects also keep track of how they were constructed, so that we can visualize them as a directed-graph.

```julia
mutable struct Value
    label::String
    data::Float64
    _prev::Set{Value}
    _op::String
end
```

The operators:

```julia
import Base.:+, Base.:*

val(label::String, x::Float64) = Value(label, x, Set(), "")

function +(a::Value, b::Value)
    return Value(
        "$(a.label) + $(b.label)",
        a.data + b.data,
        Set([a, b]),
        "+"
    )
end

function *(a::Value, b::Value)
    return Value(
        "$(a.label) * $(b.label)",
        a.data * b.data,
        Set([a, b]),
        "*"
    )
end
```

### Backpropagation

In a neural network, the end result of the value graph (root node) is the loss function $L$. The leaf nodes are the inputs that cannot change. Every value node in-between contains **weights** that can be updated during training.

We can compute the gradient at every intermediate node by working backwards from $L$. First, we can calculate the gradient of nodes that $L$ depends on directly by numeric differentiation (function `dx()`). Then, the gradient at every node before that can be computed using the chain rule.

The chain rule:

> "If a car travels twice as fast as a bicycle and the bicycle is four times as fast as a walking man, then the car travels 2 × 4 = 8 times as fast as the man.”
>
> – George F. Simmons

$$\frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx}$$

### Perceptron

The simplest “neuron” has many inputs (a vector) $\textbf{x}$  and weights $\textbf{w}$. These are multiplied together and a bias is added, then filtered through an activation function like `tanh`.

$$tanh(\textbf{w} \cdot \textbf{x} + b)$$

### Going backwards

Each op on a value node also should define a `backward` function that propagates the gradient to its children. For example the node for `a * b` defines the backwards function as:

```julia
function _backward()
		a.grad = b.data * out.grad
		b.grad = a.data * out.grad
end

```

Here `out` is the new value node that represents `a * b`.

To compute the gradient for every node, we need to call `backward()` on the nodes in the topologically-sorted order.

### A bug

The backpropagation code above has a bug. If any node is used more than once in the value graph, the gradient gets overwritten on each invocation. To fix this, we need to *accumulate* the gradient every time it is back-propagated, instead of simply setting it.

### Other operations

For convenience, we also define versions of `+` and `*` that can operate with one of the operands being a constant. We also define a few more operations: `exp` and a generalized `^` that works on integer, float and negative exponents. This also gives us division, which is just `x^-1`.

**Important**: The individual nodes in the graph can be as complex as we want. As long as the forward pass and the backward pass both work, it can be anything at all, like a “composite” operation like `tanh` or something even more complicated.

### PyTorch

The same graph and backpropagation can be done using the PyTorch API. The difference is that PyTorch operates on tensors, which are n-dimensional arrays with a rank and shape.

## Neural Net

Now we can build a neural network library on top of micrograd, closely following the PyTorch library.

A  `Neuron` has  `n`in inputs and one output: $tanh(w \cdot x + b)$.

A  `Layer` has `ni`n inputs that are fed to each of `n_out` neurons.

A  `MLP` has many layers. Each layer is fully connected to the next layer.

### Loss

The loss is a function of: (1) the vector of expected outputs (2) the vector of predicted outputs

The gradient of any parameter in the neural net with respect to the loss is a sum of terms. Each term corresponds to one training example.





