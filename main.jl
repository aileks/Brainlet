import Brainlet
using Random

function main()
    Random.seed!()
    # Random.seed!(9999)

    w1::Float64 = rand()
    w2::Float64 = rand()
    b::Float64 = rand()

    epsilon::Float64 = 1e-2
    learning_rate::Float64 = 1e-2
    epochs::Int64 = 100_000

    for _ in 1:epochs
        c::Float64 = Brainlet.cost(w1, w2, b) # this is here just to see
        println("w1 = $w1, w2 = $w2, b = $b, c = $c")
        dw1, dw2, db = Brainlet.finite_diff(w1, w2, b, epsilon)
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        b -= learning_rate * db
    end

    println("-"^80)
    predictions = Brainlet.predict(w1, w2, b)
    Brainlet.print_results(predictions)
end

main()
