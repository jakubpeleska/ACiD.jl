struct Barrier
    count::Threads.Atomic{Int}
    threshold::Int
    condvar::Condition

    function Barrier(threshold::Int)
        new(Threads.Atomic{Int}(0), threshold, Condition())
    end
end

function wait(barrier::Barrier)
    Threads.atomic_add!(barrier.count, 1)
    if barrier.count[] == barrier.threshold
        Threads.atomic_cas!(barrier.count, barrier.count[], 0)
        Threads.notify(barrier.condvar)  # Wake up all waiting processes
    else
        Threads.wait(barrier.condvar)
    end
end

function reset(barrier::Barrier)
    barrier.count = 0  # Reset
    Threads.notify(barrier.condvar)  # Wake up all waiting processes
end

function abort(barrier::Barrier)
    # TODO: error
    barrier.count = 0  # Reset
    Threads.notify(barrier.condvar)  # Wake up all waiting processes
end
