import Brainlet
using Random

function main()
    Random.seed!()
    # Random.seed!(9999)

    w::Float64 = rand() * 10
    b::Float64 = rand() * 5

    epsilon::Float64 = 1e-3
    learning_rate::Float64 = 1e-3
    epochs::Int64 = 500

    for epoch in 1:epochs
        current_cost::Float64 = Brainlet.cost(w, b)
        dw::Float64 = (Brainlet.cost(w + epsilon, b) - current_cost) / epsilon
        db::Float64 = (Brainlet.cost(w, b + epsilon) - current_cost) / epsilon
        w -= learning_rate * dw
        b -= learning_rate * db
        println("epoch = $epoch, cost = $(Brainlet.cost(w, b)), w = $w, b = $b")
    end

    println("-"^80)
    Brainlet.print_results(w, b)
end

main()
