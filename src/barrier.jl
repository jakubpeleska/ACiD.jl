mutable struct Barrier
    count::Threads.Atomic{Int}
    const threshold::Int
    condvar::Condition

    function Barrier(threshold::Int)
        new(0, threshold, ReentrantLock(), Condition())
    end
end

function wait(barrier::Barrier)
    Threads.atomic_add!(barrier.count, 1)
    if barrier.count[] == barrier.threshold
        Threads.atomic_cas!(barrier.count, barrier.count[], 0)
        notify(barrier.condvar)  # Wake up all waiting processes
    else
        wait(barrier.condvar)
    end
end

function reset(barrier::Barrier)
    barrier.count = 0  # Reset
    notify(barrier.condvar)  # Wake up all waiting processes
end

function abort(barrier::Barrier)
    # TODO: error
    barrier.count = 0  # Reset
    notify(barrier.condvar)  # Wake up all waiting processes
end
