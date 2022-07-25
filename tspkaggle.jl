# @author: Pedro Belin Castellucci
# Instituto de Informática e Estatística
# Universidade Federal de Santa Catarina

using Statistics
using Random

Random.seed!(1303)

function get_value_path(route, first, last, M)
    cost = 0.0
    for i in first:last-1
        cost += M[route[i], route[i+1]]
    end
    return cost
end


function get_value(route, M)
    total = M[route[end], route[1]]
    total += get_value_path(route, 1, length(route), M)
    total
end


function next_node(i, n)
    if i == n
        return 1
    end
    return i
end


function prev_node(i, n)
    if i == 0
        return n
    end
    return i
end


function get_edges(route)
    edges = [(route[1], route[end]), (route[end], route[1])]
    for i in 1:length(route)-1
        push!(edges, (route[i], route[i+1]))
        push!(edges, (route[i+1], route[i]))
    end
    edges
end


function generate_output(id, route, filename="submission.csv")
    edges = get_edges(route)
    line = ""
    if !isfile(filename)
        line *= "Id,Predicted\n"
    end
    n = length(route)
    for i in 1:n, j in 1:n
        if i != j
            val = 0
            if (i, j) in edges
                val = 1
            end
            line *= "$id:$(i-1)-$(j-1),$val\n"
        end
    end
    
    open(filename, "a+") do fd
        write(fd, line)
    end
end


function get_inst(idx)
    filename = "TSP2Kinput.csv"
    line = ""
    i = 0
    for ln in eachline(filename)
        if i == idx+1
            line = ln
            break;
        end
        i += 1
    end
    values = split(line, ",")
    xs = [parse(Float64, x) for x in values[2:201]]
    ys = [parse(Float64, x) for x in values[202:401]]
    ms = [parse(Int, x) for x in values[402:end]]
    x = [x for (i, x) in enumerate(xs) if ms[i] == 1]
    y = [x for (i, x) in enumerate(ys) if ms[i] == 1]
    x, y
end


function get_dist_matrix(x, y)
    n = length(x)
    dist = zeros(Float64, n, n)
    for i in 1:n
        for j in i+1:n
            dist[i, j] = sqrt((x[i] - x[j])^2 + (y[i] - y[j])^2)
            dist[j, i] = dist[i, j]
        end
    end
    dist
end


function fix_node(i, n)
    if i == 0
        return n
    end
    if i == n+1
        return 1
    end
    return i
end


function evaluate_3opt_moves(route, i, j, k, M)
    # This is based on http://tsp-basics.blogspot.com/2017/03/3-opt-move.html
    # We are only using 3 opt moves that are not 2 opt moves
    n = length(route)
    i = fix_node(i, n)
    j = fix_node(j, n)
    k = fix_node(k, n)
    a = [route[k+1:end]; route[1:i]]
    b = route[i+1:j]
    c = route[j+1:k]

    save = M[route[i], route[next_node(i, n)]]
    save += M[route[j], route[next_node(j, n)]]
    save += M[route[k], route[next_node(k, n)]]

    loss4 = M[a[end], c[1]] + M[c[end], b[1]] + M[b[end], a[1]]
    loss5 = M[a[1], b[end]] + M[b[1], c[1]] + M[c[end], a[end]]
    loss6 = M[a[1], b[1]] + M[b[end], c[end]] + M[c[1], a[end]]
    loss7 = M[a[end], c[1]] + M[c[end], b[1]] + M[b[end], a[1]]
    losses = [loss4, loss5, loss6, loss6]
    argmin = findall(==(minimum(losses)), losses)[1]
    return argmin+3, save - losses[argmin]
end


function swap_3opt!(route, i, j, k, move)
    n = length(route)
    i = fix_node(i, n)
    j = fix_node(j, n)
    k = fix_node(k, n)
    a = [route[k+1:end]; route[1:i]]
    b = route[i+1:j]
    c = route[j+1:k]
    if move == 4
        return [reverse(a); b; c]
    elseif move == 5
        return [reverse(b); reverse(b); c]
    elseif move == 6
        return [reverse(a); b; reverse(c)]
    elseif move == 7
        return [a; c; b]
    end
end


function find_best_3opt(route, M)
    n = length(route)
    best_save = 0
    best_move = best_swap = nothing
    for i in 1:n-1
        for j in i+2:n-3
            for k in j+2:n-5
                move, save = evaluate_3opt_moves(route, i, j, k, M)
                if save > best_save
                    best_save = save
                    best_move = move
                    best_swap = (i, j, k)
                end
            end
        end
    end
    if best_move != nothing
        i, j, k = best_swap
        route = swap_3opt!(route, i, j, k, move)
        return route
    end
    return nothing
end


function swap_2opt!(route, i, j)    
    if i < j
       route[i:j] = reverse(route[i:j])
    else
        route = [route[1:j]; reverse(route[j+1:i]); route[i+1:end]]
    end
    return route
end


function get_2opt_save(route, a, b, M)
    n = length(route)
    i0, i1 = a, a+1
    j0, j1 = b, b+1
    if i1 > n
        i1 = 1
    end
    if j1 > n
        j1 = 1
    end
    save = M[route[i0], route[i1]] + M[route[j0], route[j1]]
    save = save - M[route[i0], route[j0]] - M[route[i1], route[j1]]
    return save    
end


function find_best_2opt(route, M)
    n = length(route)
    best_save = 0
    best_swap = nothing
    for i in 1:n
         for j in i+2:n
            if (j+1)%n == i
                continue
            end
            
            save = get_2opt_save(route, i, j, M)
            if save > best_save
                best_save = save
                best_swap = (i+1, j)
            end
        end
    end
    if best_swap != nothing
        a, b = best_swap        
        swap_2opt!(route, a, b)
        get_value(route, M)        
        return route
    end
    return nothing
end


function hill_climbing(route, M, use_3opt)
    n = length(route)
    best = find_best_2opt(route, M)
    while best != nothing
        route = best
        best = find_best_2opt(route, M)
    end
    
    if use_3opt == true
        cand = find_best_3opt(route, M)
        while cand != nothing
            route = cand
            cand = find_best_3opt(route, m)
            @show get_value(route)
        end
    end
    @assert(length(route) == n)
    route
end


function shuffle_perturbation(route, n)
    gap = 30
    i = rand(1:n-gap)
    j = rand(i+gap:n)
    cand = [route[1:i-1]; shuffle(route[i:j]); route[j+1:end]]
    return cand
end


function remove_insert_perturbation(route, n)
    gap = 30
    i = rand(1:n-gap)
    j = rand(i+gap:n)
    section = shuffle(route[i:j])
    new_route = [route[1:i-1]; route[j+1:end]]

    if length(new_route) > 1
        k = rand(1:length(new_route))
        return [new_route[1:k]; section; new_route[k+1:end]]
    else
        return [route[1:i-1]; section; route[j+1:end]]
    end
end


function perturbation(route, n, iter)
    if iter < 400
        return shuffle_perturbation(route, n)
    elseif iter < 1500
        return remove_insert_perturbation(route, n)
    else
        return shuffle(route)
    end
end


function hill_climbing_perturbation(route, M)    
    route = hill_climbing(route, M, true)
    #print("Best after hill_climbing = $(get_value(route, M))\n")
    best_value = get_value(route, M)
    max_iter = 2500
    n = length(route)
    iter = 0
    while iter < max_iter
        cand = perturbation(route, n, iter)
        cand = hill_climbing(cand, M, false)
        cand_value = get_value(cand, M)
        if cand_value < best_value
            #println("Found best after $iter iterations = $cand_value")
            route = hill_climbing(cand, M, true)
            best_value = get_value(route, M)
            #println("Intensification with 3-opt leads to $best_value")            
            iter = 0
        end
        iter += 1
    end
    route
end


function resolve_instance(len, M)
    route = collect(1:len)
    route = hill_climbing_perturbation(route, M)
    route
end


function main()
    filename = "output_orig.csv"
    line = "Instance ID, Best solution, Time (s)\n"
    open(filename, "a+") do fd
        write(fd, line)
    end
    for id in 0:10
        print("Solving instance $id...\n")
        xs, ys = get_inst(id) 
        M = get_dist_matrix(xs, ys)
        route, time, _, _, _ = @timed resolve_instance(length(xs), M)
        best_value = get_value(route, M)
        println("Best value for instance $id = $best_value - $time seconds")
        line = "$id, $best_value, $time\n"
        open(filename, "a+") do fd
            write(fd, line)
        end
    end
end

main()