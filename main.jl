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
        dw, db = Brainlet.finite_diff(w, b, epsilon)
        w -= learning_rate * dw
        b -= learning_rate * db
        # println("epoch = $epoch, cost = $(Brainlet.cost(w, b)), w = $w, b = $b")
    end

    # println("-"^80)
    Brainlet.print_results(w, b)
end

main()
