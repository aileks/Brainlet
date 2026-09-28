module Brainlet

const TRAIN_DATA::Matrix{Float64} = [
    0 0
    1 2
    2 4
    3 6
    4 8
    5 10
]

const TRAIN_COUNT::Int64 = size(TRAIN_DATA, 1)

function finite_diff(weight::Float64, bias::Float64, epsilon::Float64)
    c::Float64 = cost(weight, bias)
    dw::Float64 = (cost(weight + epsilon, bias) - c) / epsilon
    db::Float64 = (cost(weight, bias + epsilon) - c) / epsilon
    return dw, db
end

function cost(weight::Float64, bias::Float64)
    result::Float64 = 0
    for i in 1:TRAIN_COUNT
        x::Float64 = TRAIN_DATA[i, 1]
        y::Float64 = x * weight + bias
        diff::Float64 = y - TRAIN_DATA[i, 2]
        result += diff^2
    end

    result /= TRAIN_COUNT
    return result
end

function print_results(weight::Float64, bias::Float64)
    for i in 1:TRAIN_COUNT
        x::Float64 = TRAIN_DATA[i, 1]
        actual::Float64 = x * weight + bias
        expected::Float64 = TRAIN_DATA[i, 2]
        println(actual, '\t', expected)
    end
end

end # module Brainlet
