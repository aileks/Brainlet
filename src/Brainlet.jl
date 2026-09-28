module Brainlet

# OR gate
# Columns = x1, x2, expected output
const TRAIN_DATA::Matrix{Float64} = [
    0 0 0
    1 0 1
    0 1 1
    1 1 1
]

const TRAIN_COUNT::Int64 = size(TRAIN_DATA, 1)

function cost(weight1::Float64, weight2::Float64, bias::Float64)
    result::Float64 = 0

    for i in 1:TRAIN_COUNT
        x1::Float64 = TRAIN_DATA[i, 1]
        x2::Float64 = TRAIN_DATA[i, 2]
        y::Float64 = sigmoid((x1 * weight1) + (x2 * weight2) + bias)
        diff::Float64 = y - TRAIN_DATA[i, 3]
        result += diff^2
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
function sigmoid(x::Float64)
    # TODO: Replace with ReLU
    return 1 / (1 + exp(-x))
end

function print_results(weight1::Float64, weight2::Float64, bias::Float64)
    for i in 1:TRAIN_COUNT
        x1::Float64 = TRAIN_DATA[i, 1]
        x2::Float64 = TRAIN_DATA[i, 2]
        actual::Float64 = (x1 * weight1) + (x2 * weight2) + bias
        expected::Float64 = TRAIN_DATA[i, 3]
        println(actual, '\t', expected)
    end
end

end # module Brainlet
