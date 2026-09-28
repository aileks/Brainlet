import Brainlet
using Random

function main()
    # Random.seed!()
    Random.seed!(9999)

    w1::Float64 = rand()
    w2::Float64 = rand()
    b::Float64 = rand()

    epsilon::Float64 = 1e-3
    learning_rate::Float64 = 1e-3
    epochs::Int64 = 10000

    # Test loop for viewing weights, costs, and biases as we train
    for epoch in 1:epochs
        c::Float64 = Brainlet.cost(w1, w2, b)
        println("w1 = $w1, w2 = $w2, b = $b, c = $c")
        dw1::Float64 = (Brainlet.cost(w1 + epsilon, w2, b) - c) / epsilon
        dw2::Float64 = (Brainlet.cost(w1, w2 + epsilon, b) - c) / epsilon
        db::Float64 = (Brainlet.cost(w1, w2, b + epsilon) - c) / epsilon
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        b -= learning_rate * db
    end

    # Real training
    # for epoch in 1:epochs
    #     dw1, dw2, db = Brainlet.finite_diff(w1, w2, b, epsilon)
    #     w1 -= learning_rate * dw1
    #     w2 -= learning_rate * dw2
    #     b -= learning_rate * db
    # end

    println("-"^80)
    Brainlet.print_results(w1, w2, b)
end

main()
