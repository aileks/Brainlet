module Brainlet

export NN, cost, finite_diff, print_results, backprop

#! format: off
const TRAIN_INPUTS::Matrix{Float64} = [
    0 0  0 0   # 0 + 0
    1 0  0 0   # 1 + 0
    0 1  0 0   # 2 + 0
    1 1  0 0   # 3 + 0

    0 0  1 0   # 0 + 1
    1 0  1 0   # 1 + 1
    0 1  1 0   # 2 + 1
    1 1  1 0   # 3 + 1

    0 0  0 1   # 0 + 2
    1 0  0 1   # 1 + 2
    0 1  0 1   # 2 + 2
    1 1  0 1   # 3 + 2

    0 0  1 1   # 0 + 3
    1 0  1 1   # 1 + 3
    0 1  1 1   # 2 + 3
    1 1  1 1   # 3 + 3
]

const TRAIN_TARGETS::Matrix{Float64} = [
    0 0 0   # = 0
    0 0 1   # = 1
    0 1 0   # = 2
    0 1 1   # = 3

    0 0 1   # = 1
    0 1 0   # = 2
    0 1 1   # = 3
    1 0 0   # = 4

    0 1 0   # = 2
    0 1 1   # = 3
    1 0 0   # = 4
    1 0 1   # = 5

    0 1 1   # = 3
    1 0 0   # = 4
    1 0 1   # = 5
    1 1 0   # = 6
]
#! format: on

const TRAIN_COUNT::Int64 = size(TRAIN_INPUTS, 1)

struct Layer
    weights::Matrix{Float64}
    biases::Vector{Float64}
end

struct NN
    architecture::Vector{Int64}
    layers::Vector{Layer}

    # Build each layer from the requested architecture.
    # For example, [2, 2, 1] creates a 2 -> 2 layer followed by a 2 -> 1 layer.
    function NN(architecture::Vector{Int64})
        n::Int64 = length(architecture)
        @assert n >= 2

        layers::Vector{Layer} = Layer[]

        for i in 1:(n-1)
            input_count::Int64 = architecture[i] # number of values entering this layer
            output_count::Int64 = architecture[i+1] # number of neurons in this layer

            # Set the starting weights and biases for the layer using Uniform Xavier Initialization
            # w ~ U(-x, x) where x = sqrt(6 / (n_inputs + n_outputs))
            limit = sqrt(6 / (input_count + output_count))
            weights::Matrix{Float64} = (2 .* rand(output_count, input_count) .- 1) .* limit # maps [0, 1) -> [-1, 1), then scales to [-limit, limit)
            biases::Vector{Float64} = zeros(output_count)
            push!(layers, Layer(weights, biases))
        end

        return new(architecture, layers)
    end
end

#=
The Basics

Each neuron takes some inputs, multiplies each one by a weight, adds them together with a bias,
then passes the result through an activation function (logistic sigmoid in this case):
  z = (x1 * w1) + (x2 * w2) + ... + b
  output = sigmoid(z)

Sigmoid is defined as:
   σ(z) = 1 / (1 + e⁻ᶻ)

A layer does this for every neuron it contains. The outputs from one layer become the inputs to
the next layer.

For a network with the architecture [2, 2, 1]:
  2 inputs -> 2 hidden neurons -> 1 output neuron

Prediction is therefore just repeating:
  weighted sum -> add bias -> sigmoid -> next layer


Calculating Cost/Loss

The loss for one prediction is the squared difference between the prediction and expectation:
  L = (ŷ - y)²
where ŷ is the network's prediction and y is the expected value.

The total cost is the average loss across every training example:
  C = (1/n) * Σᵢ Lᵢ
where Lᵢ is the loss for training example i and n is the number of training examples.


Finite Differences

Training requires knowing how each individual weight and bias affects the cost. For any parameter p,
its partial derivative is approximated by:
  ∂C/∂p ≈ (C(p + ε) - C(p)) / ε
where C(p) is the cost as a function of p and ε is a very small constant (e.g. 1e-3 or 0.001).

In other words, slightly increase one parameter and see how much the cost changes. This is done
separately for every weight and bias in the network.


Gradient Descent

Once the gradients are known, move each parameter in the direction that lowers the cost:
  p ← p - η * ∂C/∂p
where p can be any weight or bias in the network and η is the learning rate.
=#

# Apply sigmoid to the weighted sum plus bias in both training and prediction.
# Large positive or negative inputs saturate sigmoid and make gradients small.
function sigmoid(z::Float64)
    # TODO: Replace with ReLU
    return 1 / (1 + exp(-z))
end

# Mean squared-error loss across the whole training dataset
function cost(nn::NN)
    result::Float64 = 0
    output_count::Int = nn.architecture[end]

    for i in axes(TRAIN_INPUTS, 1)
        input, expected = sample(i)
        prediction::Vector{Float64} = forward(nn, input)

        for j in eachindex(expected)
            # L(ŷ,y)
            result += (prediction[j] - expected[j])^2
        end
    end

    return result / (TRAIN_COUNT * output_count)
end

# Using finite differences is a temporary solution for calculating gradients.
# Partial derivatives and backpropagation are not yet needed for such a small model.
function finite_diff(nn::NN, epsilon::Float64)
    # Store the completed weight and bias gradients for each layer.
    # These start empty and are filled as each layer is processed.
    weight_gradients::Vector{Matrix{Float64}} = Matrix{Float64}[]
    bias_gradients::Vector{Vector{Float64}} = Vector{Float64}[]

    for layer in nn.layers
        weight_gradient::Matrix{Float64} = zeros(size(layer.weights))
        bias_gradient::Vector{Float64} = zeros(size(layer.biases))

        # Approximate each parameter's partial derivative using a centered finite difference:
        # ∂C/∂p ≈ (C(p + ε) - C(p - ε)) / (2ε)
        # C = cost
        # p = any weight or bias
        # ε = a small constant deviation
        # This estimates how changing a parameter affects the cost without computing the derivative analytically.
        for row in axes(layer.weights, 1)
            for column in axes(layer.weights, 2)
                layer.weights[row, column] += epsilon
                cost_plus = cost(nn)
                layer.weights[row, column] -= 2 * epsilon
                cost_minus = cost(nn)
                layer.weights[row, column] += epsilon # restore original
                weight_gradient[row, column] = (cost_plus - cost_minus) / (2 * epsilon)
            end
        end

        # Do the same approximation for biases
        # layer.biases = b1, b2, ..., bn
        for i in eachindex(layer.biases)
            layer.biases[i] += epsilon
            cost_plus = cost(nn)
            layer.biases[i] -= 2 * epsilon
            cost_minus = cost(nn)
            layer.biases[i] += epsilon # restore original
            bias_gradient[i] = (cost_plus - cost_minus) / (2 * epsilon)
        end

        push!(weight_gradients, weight_gradient)
        push!(bias_gradients, bias_gradient)
    end

    return weight_gradients, bias_gradients
end

#=
What is Backpropagation?

During the forward pass, each layer computes:
  z = w*aₚ + b
  a = σ(z)
where:
  aₚ = activation vector from the previous layer
  w  = weight matrix
  b  = bias vector
  z  = pre-activation vector
  a  = output activation vector
  σ  = activation function

Backpropagation computes how much each weight and bias contributed to the final loss L by applying
the chain rule, working backward from the output layer toward the input layer.

Each layer has an error signal vector:
  δ = ∂L/∂z

For the output layer, δ is computed directly from the derivative of the loss and the derivative of
the activation function.

For a hidden layer:
  δ⁽ˡ⁾ = (w⁽ˡ⁺¹⁾)ᵀ * δ⁽ˡ⁺¹⁾ ⊙ σ'(z⁽ˡ⁾)
where ˡ is the layer index and ⊙ is elementwise multiplication.

In other words, the next layer's error is propagated backward through its weights, then scaled
elementwise by this layer's activation derivative. Effectively, this is the network working backward
to figure out where the mistake came from and how strongly each part should be adjusted.

Once δ is known, the gradients are:
  ∂L/∂w = δ * aₚᵀ
  ∂L/∂b = δ

For an individual weight:
  ∂L/∂wᵢⱼ = δᵢ * aₚⱼ

So, each weight's gradient is the error of the neuron it feeds into, multiplied by the activation
that passed through that weight.

Gradient descent then updates the parameters:
  w ← w - η * δ * aₚᵀ
  b ← b - η * δ
where η is the learning rate.

In short:
  forward:   aₚ → z → a → L
  backward:  L → δ → ∂L/∂w, ∂L/∂b
  update:    parameters ← parameters - η * gradients
=#
function backprop(nn::NN)
    # Accumulate ∂L/∂W and ∂L/∂b for every layer.
    weight_gradients::Vector{Matrix{Float64}} = [zeros(size(layer.weights)) for layer in nn.layers]
    bias_gradients::Vector{Vector{Float64}} = [zeros(size(layer.biases)) for layer in nn.layers]

    for i in axes(TRAIN_INPUTS, 1)
        input, expected = sample(i)

        # Cache the values produced during the forward pass because backprop will need them while moving backward.
        activations::Vector{Vector{Float64}} = [input]
        zs::Vector{Vector{Float64}} = Vector{Float64}[] # the vectors of pre-activations
        input_a::Vector{Float64} = input # the activations from the previous layer

        # Do a forward pass through each layer and store pre-activations and activations for use during backprop.
        for layer in nn.layers
            z::Vector{Float64} = layer.weights * input_a + layer.biases
            input_a = sigmoid.(z)

            push!(zs, z)
            push!(activations, input_a)
        end

        # Compute δ for the output layer.
        #    ∂L/∂a = 2(a-y)
        #    a = σ(z)
        #    σ'(z) = e^(-z)/(1+e^(-z))^2 = a(1-a)
        #    δ = ∂L/∂z
        #      = ∂L/∂a * ∂a/∂z
        #      = 2(a-y) * σ'(z)
        #      = 2(a-y) * a(1-a)
        # `y` is the `expected` vector here.
        output_a::Vector{Float64} = activations[end]
        loss_gradient::Vector{Float64} = 2 .* (output_a .- expected)
        delta::Vector{Float64} = loss_gradient .* output_a .* (1 .- output_a)

        # Use δ to compute the output layer's ∂L/∂W and ∂L/∂b.
        #    ∂L/∂W = δ * aₚᵀ
        #    ∂L/∂b = δ
        # where aₚ is the activation vector from the previous layer.
        prev_a::Vector{Float64} = activations[end-1]
        weight_gradients[end] .+= delta * transpose(prev_a)
        bias_gradients[end] .+= delta

        # Propagate δ backward through each hidden layer.
        #    ∂L/∂a⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ * δ⁽ˡ⁺¹⁾
        #    δ⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ = ∂L/∂a⁽ˡ⁾ ⊙ σ'(z⁽ˡ⁾)
        #    σ'(z⁽ˡ⁾) = a⁽ˡ⁾ ⊙ (1 - a⁽ˡ⁾)
        #    δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ * δ⁽ˡ⁺¹⁾ ⊙ a⁽ˡ⁾ ⊙ (1 - a⁽ˡ⁾)
        for j in (length(nn.layers)-1):-1:1
            curr::Vector{Float64} = activations[j+1]
            prev::Vector{Float64} = activations[j]
            w::Matrix{Float64} = nn.layers[j+1].weights
            delta = transpose(w) * delta .* curr .* (1 .- curr)
            weight_gradients[j] .+= delta * transpose(prev)
            bias_gradients[j] .+= delta
        end
    end

    # Convert the accumulated per-example gradients into gradients of the mean cost used by cost(nn).
    # NOTE: Perhaps rewrite the cost function as a summed squared error to avoid this loop?
    n = TRAIN_COUNT * nn.architecture[end]
    for i in eachindex(weight_gradients)
        weight_gradients[i] ./= n
        bias_gradients[i] ./= n
    end

    return weight_gradients, bias_gradients
end

function sample(i::Int)
    input::Vector{Float64} = TRAIN_INPUTS[i, :]
    expected::Vector{Float64} = TRAIN_TARGETS[i, :]

    return input, expected
end

# Get predictions of the network.
function forward(nn::NN, input::Vector{Float64})
    result::Vector{Float64} = input

    for layer in nn.layers
        result = sigmoid.(layer.weights * result + layer.biases)
    end

    return result
end

# Debug function to take a peak at hidden layeres
function layer_debug(nn::NN, input::Vector{Float64})
    result::Vector{Float64} = input

    for (i, layer) in enumerate(nn.layers)
        result = sigmoid.(layer.weights * result + layer.biases)
        println("layer $i: $result")
    end

    return result
end

function print_results(nn::NN; digits::Int=3)
    headers = ("Input", "Prediction", "Expected")
    rows = Tuple{String,String,String}[]

    format_vector(v) = "[" * join(string.(round.(v; digits=digits)), ", ") * "]"

    for i in axes(TRAIN_INPUTS, 1)
        input, expected = sample(i)
        prediction = forward(nn, input)

        push!(rows, (
            format_vector(input),
            format_vector(prediction),
            format_vector(expected),
        ))
    end

    widths = collect(length.(headers))

    for row in rows
        for j in eachindex(widths)
            widths[j] = max(widths[j], length(row[j]))
        end
    end

    function print_row(row)
        println(
            rpad(row[1], widths[1]), " | ",
            rpad(row[2], widths[2]), " | ",
            row[3],
        )
    end

    print_row(headers)
    println(join(("-"^width for width in widths), "-+-"))

    for row in rows
        print_row(row)
    end
end

end # module Brainlet
