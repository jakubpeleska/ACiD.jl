
# import os
# import torch
# import random
# import numpy as np
# import torch.distributed as dist
# import multiprocessing as mp
# from multiprocessing import Process, Manager
# from utils.graph_utils import ExponentialGraph, CycleGraph

# functions need to be exported for the documentation
export sync_process, listen_given_rank, master_process

const KILL_PROCESS_SIGNAL = Int64(-2)

const AVAILABLE_TAG = 123
const RANK_TAG = 234



"""
### Process run by every worker in the background.
This process allows each worker to communicate with the "orchestrating" master process (hosted by worker 0).
The goal is to signal to the master process when this worker is available for communication,
and to gather from the master process the rank of the peer with which we are supposed to communicate.
When received, this information `rank_other` is sent to the p2p\\_averaging process run in parallel in this worker,
so that the p2p averaging process knows with which worker to communicate next.
This process will communicate with one of the `world_size` "listen\\_given\\_rank" processes hosted at worker 0, which has `world_size + 1` processes run in parallel:
    * one to `listen_to` each one of the sync_process run by each worker.
    * one "orchestrating" process, dedicated to make pairs of workers.
So, in total, there are `2*world_size + 1` processes that need to communicate with each other (only sending ints), so initialize a process\\_group using gloo backend here.

Parameters:
- rank (int): our rank id in the distributed setting.
- world\\_size (int): the total number of workers.
- rank\\_other (mp.Value): a multiprocessing Value to store the id of the rank of the next communication.
                            It is updated here, based on the information given by the master process, to signal to the p2p\\_averaging process
                            run in parallel in this worker which peer to communicate wiith next.
                            if `rank_other.value` == -2: signal from the orchestrating process that enough gradients have been computed in total,
                                                    stops the communication process.
- new\\_grads (mp.Value): a multiprocessing Value updated by the process and the main one, counting how many new grad steps have been performed
                        by this worker since last communication. This is used by the master process to count the total number of grad done,
                        and initiate the "kill" of all processes when the right amount of grad steps have been performed in total.
- barrier\\_sync\\_averaging (mp.Barrier): a barrier used to communicate with the p2p\\_averaging process.
                                        When the averaging process meets this barrier, it signals to this process that the worker
                                        is available for the next communication, so we can begin to look for another available peer to connect
                                        to by sending our rank information to the master process which will realize the pairing.
- log (logger): to print messages in the logs if needed.
"""
function sync_process(
    comm::MPI.Comm,
    master_rank::Int64,
    other_rank::Threads.Atomic{Int64},
    ∇new::Threads.Atomic{Int64},
    barrier_sync_averaging::Barrier,
)

    while true
        ∇newₙ = ∇new[]

        # send a tensor to master to signal worker is available to communicate
        MPI.send(∇newₙ, comm, dest = master_rank, tag = AVAILABLE_TAG)

        # re-initialize the new_grads value
        Threads.atomic_sub!(∇new, ∇newₙ)

        # receive the rank from the last process in the group
        other_rankₙ = MPI.recv(comm, source = master_rank, tag = RANK_TAG)

        # changes the rank value localy saved in the mp.Value variable
        Threads.atomic_cas!(other_rank, other_rank[], other_rankₙ)
        if other_rankₙ == KILL_PROCESS_SIGNAL
            # signal to the listening process to kill the process
            MPI.send(
                KILL_PROCESS_SIGNAL,
                comm,
                dest = master_rank,
                tag = AVAILABLE_TAG,
            )

            abort(barrier_sync_averaging)
            break
        end

        # wait for the p2p averaging
        wait(barrier_sync_averaging)
        reset(barrier_sync_averaging)
    end
end

"""
### Process run in the background of worker 0.
Its goal is to listen to one specific worker (specifically, its "sync\\_process" process), and to send it information coming from the orchestrating process also hosted by worker 0.
The main goal of this function is to put to the mp.Queue the rank of the worker it is listening to when this worker sent, through its "sync\\_process" function, the signal that its
corresponding worker was available for a communication.
Then, as this mp.Queue is shared with the orchestrating process, the orchestrating process can receive the information and pair the worker with another one.

Parameters:
- rank (int): our rank id in the distributed setting.
- world\\_size (int): the total number of workers.
- queue (mp.Queue): queue containing the ranks of all available workers for communication.
                    The orchestrating process then only needs to "de-queue" the ranks to make pairs, insuring that the communications are performed in FIFO style,
                    minimizing latency.
- nb\\_grad\\_tot\\_so\\_far (mp.Value): int storing the global count of grads (total number of gradients taken by all workers).
                                    This value is updated by adding to it the "new\\_grads" (see "sync\\_process" doc) from every worker.
                                    This mp.Value is thus updated by world\\_size "listen\\_given\\_rank" processes, and used by the orchestrating process to kill all processes
                                    when the target number of grads is reached.
- lock (mp.Lock): multiprocessing lock to make sure that the nb\\_grad\\_tot\\_so\\_far is edited by only one process at a time, so that no "new gradients" are thrown out
                    by a multiprocessing bug.
- log (logger): to print messages in the logs if needed.
"""
function listen_given_rank(
    comm::MPI.Comm,
    rank::Int64,
    queue::Channel{Int64},
    ∇steps::Threads.Atomic{Int64},
)
    while true
        # receive information that worker rank is available for communications.
        # the act of receiving a message is the signal itself.
        # the inside of 'tensor_other_rank' variable contains the "new grads" performed by rank since last communication.
        ∇new = MPI.recv(comm, source = rank, tag = AVAILABLE_TAG)

        # order to kill all processes
        if ∇new == KILL_PROCESS_SIGNAL
            break
        end

        # add the new grads
        Threads.atomic_add!(∇steps, ∇new)
        # signal the orchestrating process that worker nb rank is available for communication
        put!(queue, rank)
    end
end


"""
### Orchestrating process hosted on worker 0.
This process accomplishes 2 things:
* Group available workers by pairs for p2p communication, according to the given graph topology, and trying to minimize latency
    by pairing together workers that were the first to be available to communicate.
* Signal to all processes when the target number of grads have been reached, so that computations & communication can stop.

Parameters:
- world\\_size (int): the total number of workers.
- ∇max\\_steps (int): The target number of total nb of grads performed by all workers.
                            When it is reached, this process sends the signal to all other to stop all computations & communications.
"""
function master_process(
    comm::MPI.Comm,
    world_size::Int64,
    should_run::Threads.Atomic{Bool};
    ∇max_steps::Int64 = -1,
)
    # Initialize multiprocessing variables shared with the "listen_given_rank" processes, also hosted by worker 0.
    # Queue containing the rank of available workers. 
    # Ranks are enqueued by their corresponding "listen_given_rank" process, and dequeued here to make pairs.
    queue = Channel{Int64}(Inf)

    # Init an int storing the global count of grads (total number of gradients taken by all workers).
    # this mp.Value is shared with all "listen_given_rank" processes, so that they can edit it.
    # When the target number of grads is reached, this process sends the signal to all other to stop all computations & communications.
    ∇steps = Threads.Atomic{Int64}(0)

    # launch the listening processes for each rank
    list_processes = [
        Threads.@spawn listen_given_rank(comm, rank, queue, ∇steps) for
        rank in 1:world_size
    ]

    # tuple of ranks stores the first 2 available workers to communicate
    ranks = Vector{Int64}()

    # while the total number of grad is not reached
    while should_run[] && (∇max_steps == -1 || ∇steps[] < ∇max_steps)
        # get the rank of the first available worker
        push!(ranks, take!(queue))
        # if 2 workers are available for communication
        if length(ranks) == 2
            # send their ranks to each other
            MPI.send(ranks[1], comm, dest = ranks[2])
            MPI.send(ranks[2], comm, dest = ranks[1])
            # re-initialize the tuple as an empty one
            ranks = Vector{Int64}()
        end
    end

    # when we go out of the while loop, send to everybody the message to stop processes
    req = [
        MPI.Isend(KILL_PROCESS_SIGNAL, comm, dest = rank) for
        rank in 0:world_size
    ]

    MPI.Waitall(req)

    # terminates all processes
    fetch.(list_processes)
end

