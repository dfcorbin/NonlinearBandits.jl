"""
    RandomPolicy(num_actions::Int64)

Construct a policy that chooses actions at random.
"""
struct RandomPolicy <: AbstractPolicy
    num_actions::Int64
    function RandomPolicy(num_actions::Int64)
        return if num_actions > 0
            new(num_actions)
        else
            throw(ArgumentError("num_actions must be positive"))
        end
    end
end

function (pol::RandomPolicy)(X::AbstractMatrix{Float64})
    n = size(X, 2)
    return rand(1:(pol.num_actions), n)
end
