using Brainlet
using Random

function main()
    Random.seed!()
    # Random.seed!(9999)

    arch::Vector{Int} = [2, 2, 1]
    nn = NN(arch)

    epsilon::Float64 = 1e-2
    learning_rate::Float64 = 1e-1
    epochs::Int64 = 100_000

    for _ in 1:epochs
        c::Float64 = cost(nn) # this is here just to see
        println("cost = $c")
        weight_gradients, bias_gradients = finite_diff(nn, epsilon)

        # Apply gradients to current layer's weights and biases
        # W = current weights
        # b = current biases
        # η = learning rate
        # C = cost
        for (layer, weight_gradient, bias_gradient) in zip(nn.layers, weight_gradients, bias_gradients)
            # W = W - η * (∂C/∂W)
            layer.weights .-= learning_rate .* weight_gradient
            # b = b - η * (∂C/∂b)
            layer.biases .-= learning_rate .* bias_gradient
        end
    end

    println("-"^50)
    print_results(nn)
end

main()
