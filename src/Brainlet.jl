module Brainlet

export NN, cost, finite_diff, predict, print_results

#=
# XOR cannot actually be modeled with a single neuron, even with many inputs.
#
# A single neuron can only create one decision boundary. XOR needs to separate
# (0, 1) and (1, 0) from both (0, 0) and (1, 1), which cannot be done with
# a single straight line.
#
# In other words, no amount of training is going to fix this. The model simply
# isn't capable of representing the function we're asking it to learn.
#
# It's time to make the model more complex.
=#

# XOR gate
# Columns = x1, x2, expected output
const TRAIN_DATA::Matrix{Float64} = [
    0 0 0
    1 0 1
    0 1 1
    1 1 0
]

const TRAIN_COUNT::Int64 = size(TRAIN_DATA, 1)

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

            # Initialize the starting weights and biases for the layer
            weights::Matrix{Float64} = rand(output_count, input_count)
            biases::Vector{Float64} = rand(output_count)
            push!(layers, Layer(weights, biases))
        end

        return new(architecture, layers)
    end
end

#=
# How the Math Works
#
# Each neuron takes some inputs, multiplies each one by a weight, adds them together with a bias,
# then passes the result through an activation function (logistic sigmoid in this case):
#   z = (x1 * w1) + (x2 * w2) + ... + b
#   output = sigmoid(z)
#
# Sigmoid being defined as:
#    sigmoid(z) = 1 / (1 + e^(-z))
#
# A layer does this for every neuron it contains. The outputs from one layer become the inputs to
# the next layer.
#
# For a network with the architecture [2, 2, 1]:
#   2 inputs -> 2 hidden neurons -> 1 output neuron
#
# Prediction is therefore just repeating:
#   weighted sum -> add bias -> sigmoid -> next layer
#
#
# Calculating Cost/Loss
#
# The loss for one prediction is the squared difference between what the network predicted and what
# we expected:
#   loss = (prediction - expected)^2
#
# The total cost is the average loss across every training example:
#   cost = sum(losses) / number of examples
#
#
# Finite Differences
#
# Training requires knowing how each individual weight and bias affects the cost. For any parameter p,
# its partial derivative is approximated by:
#   gradient = (cost(p + epsilon) - cost(p)) / epsilon
#
# In other words, slightly increase one parameter and see how much the cost changes. This is done
# separately for every weight and bias in the network.
#
#
# Gradient Descent
#
# Once the gradients are known, move each parameter in the direction that lowers the cost:
#   p = p - (learning_rate * gradient)
# where p can be any weight or bias in the network.
=#

# Mean squared-error loss across the whole training dataset
function cost(nn::NN)
    result::Float64 = 0

    for i in 1:TRAIN_COUNT
        input::Vector{Float64} = TRAIN_DATA[i, 1:2]
        expected::Float64 = TRAIN_DATA[i, 3]
        prediction::Float64 = predict(nn, input)[1]
        loss::Float64 = (prediction - expected)^2
        result += loss
    end

    result /= TRAIN_COUNT
    return result
end

# Using finite differences is a temporary solution for calculating gradients.
# Partial derivatives and backpropagation are not yet needed for such a small model.
function finite_diff(nn::NN, epsilon::Float64)
    c::Float64 = cost(nn)

    # Store the completed weight and bias gradients for each layer.
    # These start empty and are filled as each layer is processed.
    weight_gradients::Vector{Matrix{Float64}} = Matrix{Float64}[]
    bias_gradients::Vector{Vector{Float64}} = Vector{Float64}[]

    for layer in nn.layers
        weight_gradient::Matrix{Float64} = zeros(size(layer.weights))
        bias_gradient::Vector{Float64} = zeros(size(layer.biases))

        # Approximate each weight's partial derivative layer.weights = w1, w2, ..., wn
        for row in axes(layer.weights, 1)
            for column in axes(layer.weights, 2)
                layer.weights[row, column] += epsilon # w1 + epsilon (and w2 + epsilon) in the AND/OR nn
                # approximation of ∂C/∂w
                weight_gradient[row, column] = (cost(nn) - c) / epsilon # dw1/dw2 in the AND/OR nn
                layer.weights[row, column] -= epsilon
            end
        end

        # Do the same approximation for biases
        # layer.biases = b1, b2, ..., bn
        for i in eachindex(layer.biases)
            layer.biases[i] += epsilon # b + epsilon in the AND/OR nn
            # approximation of ∂C/∂b
            bias_gradient[i] = (cost(nn) - c) / epsilon # db in the AND/OR nn
            layer.biases[i] -= epsilon
        end

        push!(weight_gradients, weight_gradient)
        push!(bias_gradients, bias_gradient)
    end

    return weight_gradients, bias_gradients
end

# Apply sigmoid to the weighted sum plus bias in both training and prediction.
# Large positive or negative inputs saturate sigmoid and make gradients small.
function sigmoid(x::Float64)
    # TODO: Replace with ReLU
    return 1 / (1 + exp(-x))
end

# Get predictions of the network.
function predict(nn::NN, input::Vector{Float64})
    result::Vector{Float64} = input

    for layer in nn.layers
        result = sigmoid.(layer.weights * result + layer.biases)
    end

    return result
end

function print_results(nn::NN)
    for (x1, x2, expected) in eachrow(TRAIN_DATA)
        input::Vector{Float64} = [x1, x2]
        prediction::Float64 = round(predict(nn, input)[1]; digits=6)
        println("$(Int(x1)) | $(Int(x2)) -> $prediction :: Expected $(Int64(expected))")
    end
end

end # module Brainlet
