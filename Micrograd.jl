module Micrograd

export val, +, *, tanh, exp
export Value, Neuron, Layer, MultiLayerPerceptron
export visualize
export backpropagate
export neuron, layer, mlp, forward, parameters

import Base.:+, Base.:-, Base.:*, Base.^, Base.:tanh, Base.:exp
import GraphViz

mutable struct Value
    label::String
    data::Float64
    grad::Float64
    _prev::Set{Value}
    _op::String
    _backward::Function
end

val(label::String, x::Float64) = Value(label, x, 0.0, Set(), "", () -> nothing)

Base.show(io::IO, v::Value) = print(io, "Value($(v.data))")

function +(a::Value, b::Value)
    out = Value(
        "",
        a.data + b.data,
        0.0,
        Set([a, b]),
        "+",
        () -> nothing
    )

    function _backward()
        a.grad += 1.0 * out.grad
        b.grad += 1.0 * out.grad
    end
    out._backward = _backward

    return out
end
+(a::Value, b::Float64) = a + val("const", b)
+(a::Float64, b::Value) = val("const", a) + b

function -(a::Value, b::Value)
    out = Value(
        "",
        a.data - b.data,
        0.0,
        Set([a, b]),
        "-",
        () -> nothing
    )

    function _backward()
        a.grad += 1.0 * out.grad
        b.grad += -1.0 * out.grad
    end
    out._backward = _backward

    return out
end
-(a::Value, b::Float64) = a - val("const", b)
-(a::Float64, b::Value) = val("const", a) - b


function *(a::Value, b::Value)
    out = Value(
        "",
        a.data * b.data,
        0.0,
        Set([a, b]),
        "*",
        () -> nothing
    )

    function _backward()
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad
    end
    out._backward = _backward

    return out
end
*(a::Value, b::Float64) = a * val("const", b)
*(a::Float64, b::Value) = val("const", a) * b

function ^(a::Value, b::Float64)
    out = Value(
        "",
        a.data ^ b,
        0.0,
        Set([a]),
        "^",
        () -> nothing
    )

    function _backward()
        a.grad += b * a.data^(b - 1.0) * out.grad
    end
    out._backward = _backward

    return out
end

function tanh(v::Value)
    out = Value(
        "",
        tanh(v.data),
        0.0,
        Set([v]),
        "tanh",
        () -> nothing
    )

    function _backward()
        v.grad += (1.0 - tanh(v.data)^2) * out.grad
    end
    out._backward = _backward

    return out
end

function exp(v::Value)
    out = Value(
        "",
        exp(v.data),
        0.0,
        Set([v]),
        "exp",
        () -> nothing
    )

    function _backward()
        v.grad += exp(v.data) * out.grad
    end
    out._backward = _backward

    return out
end


## Visualization
function trace(root::Value)
    nodes, edges = Set(), Set()

    function build(v::Value)
        if v ∉ nodes
            push!(nodes, v)
            for child in v._prev
                push!(edges, (child, v))
                build(child)
            end
        end
    end
    build(root)

    return nodes, edges
end

function dot_str(v::Value)::String
    nodes, edges = trace(v)
    lines = String[]

    for node in nodes
        uid = string(hash(node))
        push!(lines, """"$(uid)" [label="data = $(node.data) | grad = $(node.grad)"; shape="record"]""")

        if node._op != ""
            push!(lines, """ "$(uid * node._op)" [label="$(node._op)"]""")
            push!(lines, """ "$(uid * node._op)" -> "$(uid)" """)
        end
    end
    for (a, b) in edges
        push!(lines, """ "$(string(hash(a)))" -> "$(string(hash(b)) * b._op)" """)
    end
    attrs = [
        "graph [rankdir=LR;]"
    ]
    return "digraph G {\n\t" * join(attrs, "\n\t") * "\n" * join(lines, "\n\t") * "\n}"
end

function visualize(v::Value)
    g = GraphViz.Graph(dot_str(v))
    GraphViz.layout!(g, engine="dot")
    return g
end

## Backpropagation
function toposort(root::Value)
    visited = Set()
    order = Value[]

    function visit(v::Value)
        if v ∉ visited
            for child in v._prev
                visit(child)
            end
            push!(order, v)
            push!(visited, v)
        end
    end
    visit(root)

    return reverse(order)
end

function backpropagate(root::Value)
    for node in toposort(root)
        node._backward()
    end
end


struct Neuron
    nin::Integer
    w::Array{Value,1}
    b::Value
end

function neuron(nin::Integer)
    w = [val("w$i", rand()) for i in 1:nin]
    b = val("b", rand())
    return Neuron(nin, w, b)
end

function forward(n::Neuron, x::Array{Value,1})
    return tanh(sum(n.w .* x) + n.b)
end
forward(n::Neuron, x::Array{Float64, 1}) = forward(n, [val("x$i", x[i]) for i in 1:length(x)])

parameters(n::Neuron) = [n.w; n.b]

struct Layer
    neurons::Array{Neuron,1}
end

function layer(nin::Integer, nout::Integer)
    neurons = [neuron(nin) for i in 1:nout]
    return Layer(neurons)
end

# Call parameters on every neuron and flatten the result
parameters(l::Layer) = reduce(vcat, [parameters(n) for n in l.neurons])

function forward(l::Layer, x::Array{Value,1})
    return [forward(n, x) for n in l.neurons]
end
forward(l::Layer, x::Array{Float64, 1}) = forward(l, [val("x$i", x[i]) for i in 1:length(x)])

struct MultiLayerPerceptron
    layers::Array{Layer,1}
end

function mlp(nin::Int64, nouts::Vector{Int64})
    # nin is the number of inputs to the first layer
    # nouts is the number of neurons in each subsequent layer

    sizes = [nin; nouts]
    layers = [layer(sizes[i], sizes[i+1]) for i in 1:length(sizes)-1]
    return MultiLayerPerceptron(layers)
end

function forward(m::MultiLayerPerceptron, x::Array{Value,1})
    for l in m.layers
        x = forward(l, x)
    end
    return x
end

forward(m::MultiLayerPerceptron, x::Array{Float64, 1}) = forward(m, [val("x$i", x[i]) for i in 1:length(x)])

parameters(m::MultiLayerPerceptron) = reduce(vcat, [parameters(l) for l in m.layers])

end