using Brainlet
using Random

const BITS::Int = 4

function main()
    # Random.seed!()
    Random.seed!(9999)

    arch::Vector{Int} = [BITS, 2BITS, 2BITS, 2BITS, BITS - 1]
    nn::NN = NN(arch)

    learning_rate::Float64 = 1e-1
    epochs::Int64 = 500_000

    println("First cost: $(cost(nn))")
    for _ in 1:epochs
        bp_w, bp_b = backprop(nn)

        for (layer, weight_gradient, bias_gradient) in zip(nn.layers, bp_w, bp_b)
            layer.weights .-= learning_rate .* weight_gradient
            layer.biases .-= learning_rate .* bias_gradient
        end
    end

    println("Final cost: $(cost(nn))")
    println("-"^50)
    print_results(nn)
end

main()
