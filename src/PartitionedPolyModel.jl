mutable struct Region
    key::Int64
    limits::Matrix{Float64}
    d::Int64
    loc::Float64
    left::Region
    right::Region

    function Region(key::Int64, limits::Matrix{Float64})
        return new(key, limits)
    end
end


mutable struct Partition
    space::Region
    regions::Vector{Region}
end


function Partition(limits::Matrix{Float64})
    space = Region(1, limits)
    return Partition(space, [space])
end


function split!(prt::Partition, idx::Int64, d::Int64)
    region = prt.regions[idx]
    loc = sum(region.limits[d, :]) * 0.5
    left_limits = deepcopy(region.limits)
    right_limits = deepcopy(region.limits)
    left_limits[d, 2] = right_limits[d, 1] = loc

    region.key = 0
    region.d = d
    region.loc = loc
    region.left = Region(idx, left_limits)
    region.right = Region(length(prt.regions) + 1, right_limits)

    prt.regions[idx] = region.left
    push!(prt.regions, region.right)
    return nothing
end


function locate(region::Region, x::AbstractVector)
    if region.key != 0
        return region.key
    end
    if x[region.d] < region.loc
        locate(region.left, x)
    else
        locate(region.right, x)
    end
end


function locate(prt::Partition, X::Matrix{Float64})
    return [locate(prt.space, x) for x in eachrow(X)]
end