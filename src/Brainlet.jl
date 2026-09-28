module Brainlet

# NAND gate
# Columns = x1, x2, expected output
const TRAIN_DATA::Matrix{Float64} = [
    0 0 1
    1 0 1
    0 1 1
    1 1 0
]

const TRAIN_COUNT::Int64 = size(TRAIN_DATA, 1)

# Mean squared-error loss across the whole training dataset
function cost(weight1::Float64, weight2::Float64, bias::Float64)
    result::Float64 = 0

    for i in 1:TRAIN_COUNT
        x1::Float64 = TRAIN_DATA[i, 1]
        x2::Float64 = TRAIN_DATA[i, 2]
        y::Float64 = sigmoid((x1 * weight1) + (x2 * weight2) + bias)
        loss::Float64 = (y - TRAIN_DATA[i, 3])^2
        result += loss
    end

    result /= TRAIN_COUNT
    return result
end

# Using finite differences is a temporary solution for calculating gradients.
# Partial derivatives and backpropagation are not yet needed for such a small model.
function finite_diff(weight1::Float64, weight2::Float64, bias::Float64, epsilon::Float64)
    # TODO: Replace with backpropagation
    c::Float64 = cost(weight1, weight2, bias)
    dw1::Float64 = (cost(weight1 + epsilon, weight2, bias) - c) / epsilon
    dw2::Float64 = (cost(weight1, weight2 + epsilon, bias) - c) / epsilon
    db::Float64 = (cost(weight1, weight2, bias + epsilon) - c) / epsilon
    return dw1, dw2, db
end

# Logistic sigmoid function; good enough for current needs.
# More complexity may require ReLU in the future.
# Apply sigmoid to the weighted sum plus bias in both training and prediction.
# Large positive or negative inputs saturate sigmoid and make gradients small.
function sigmoid(x::Float64)
    # TODO: Replace with ReLU
    return 1 / (1 + exp(-x))
end

function predict(weight1::Float64, weight2::Float64, bias::Float64)
    return sigmoid.(TRAIN_DATA[:, 1] .* weight1 .+ TRAIN_DATA[:, 2] .* weight2 .+ bias)
end

function print_results(predictions::Vector{Float64})
    for (i, (x1, x2, expected)) in enumerate(eachrow(TRAIN_DATA))
        prediction = round(predictions[i]; digits=6)
        println("$(Int(x1)) | $(Int(x2)) -> $prediction :: Expected $(Int64(expected))")
    end
end

end # module Brainlet
