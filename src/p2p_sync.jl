
# import os
# import torch
# import random
# import numpy as np
# import torch.distributed as dist
# import multiprocessing as mp
# from multiprocessing import Process, Manager
# from utils.graph_utils import ExponentialGraph, CycleGraph


"""
### Process run by every worker in the background.
This process allows each worker to communicate with the "orchestrating" master process (hosted by worker 0).
The goal is to signal to the master process when this worker is available for communication,
and to gather from the master process the rank of the peer with which we are supposed to communicate.
When received, this information "rank_other" is sent to the p2p_averaging process run in parallel in this worker,
so that the p2p averaging process knows with which worker to communicate next.
This process will communicate with one of the world_size "listen_given_rank" processes hosted at worker 0, which has world_size + 1 processes run in parallel:
    * one to "listen_to" each one of the sync_process run by each worker.
    * one "orchestrating" process, dedicated to make pairs of workers.
So, in total, there are 2*world_size + 1 processes that need to communicate with each other (only sending ints), so initialize a process_group using gloo backend here.

Parameters:
- rank (int): our rank id in the distributed setting.
- world_size (int): the total number of workers.
- rank_other (mp.Value): a multiprocessing Value to store the id of the rank of the next communication.
                            It is updated here, based on the information given by the master process, to signal to the p2p_averaging process
                            run in parallel in this worker which peer to communicate wiith next.
                            if rank_other.value == -2: signal from the orchestrating process that enough gradients have been computed in total,
                                                    stops the communication process.
- new_grads (mp.Value): a multiprocessing Value updated by the process and the main one, counting how many new grad steps have been performed
                        by this worker since last communication. This is used by the master process to count the total number of grad done,
                        and initiate the "kill" of all processes when the right amount of grad steps have been performed in total.
- barrier_sync_averaging (mp.Barrier): a barrier used to communicate with the p2p_averaging process.
                                        When the averaging process meets this barrier, it signals to this process that the worker
                                        is available for the next communication, so we can begin to look for another available peer to connect
                                        to by sending our rank information to the master process which will realize the pairing.
- log (logger): to print messages in the logs if needed.
"""
function sync_process(
    rank,
    world_size,
    rank_other,
    new_grads,
    barrier_sync_averaging,
    log,
)

    # initialize the process group for rank communications
    process_group = dist.init_process_group(
        backend = "gloo",
        init_method = "tcp://" +
                      os.environ["MASTER_ADDR"] +
                      ":" +
                      str(int(os.environ["MASTER_PORT"]) + 1),
        rank = rank,
        world_size = 2 * world_size + 1,
    )

    while true
        # initialize a tensor to send master process
        # use the number of grad steps done by worker rank since last communication as message
        # we use the same tensor as placeholder to receive the other rank
        tensor_other_rank = torch.ones(1) * new_grads.value
        # send a tensor to master to signal worker nb rank is available to communicate
        dist.send(tensor_other_rank, rank + world_size, process_group)
        # re-initialize the new_grads value
        new_grads.value = new_grads.value - int(tensor_other_rank.data)
        # receive the rank from the last process in the group
        dist.recv(tensor_other_rank, 2 * world_size, process_group)
        # changes the rank value localy saved in the mp.Value variable
        rank_other.value = int(tensor_other_rank.data)
        if rank_other.value == -2
            # signal to the listening process to kil the process
            dist.send(tensor_other_rank, rank + world_size, process_group)
            barrier_sync_averaging.abort()
            break
        end
        # wait for the p2p averaging
        barrier_sync_averaging.wait()
        barrier_sync_averaging.reset()
    end
end

"""
### Process run in the background of worker 0.
Its goal is to listen to one specific worker (specifically, its "sync_process" process), and to send it information coming from the orchestrating process also hosted by worker 0.
The main goal of this function is to put to the mp.Queue the rank of the worker it is listening to when this worker sent, through its "sync_process" function, the signal that its
corresponding worker was available for a communication.
Then, as this mp.Queue is shared with the orchestrating process, the orchestrating process can receive the information and pair the worker with another one.

Parameters:
- rank (int): our rank id in the distributed setting.
- world_size (int): the total number of workers.
- queue (mp.Queue): queue containing the ranks of all available workers for communication.
                    The orchestrating process then only needs to "de-queue" the ranks to make pairs, insuring that the communications are performed in FIFO style,
                    minimizing latency.
- nb_grad_tot_so_far (mp.Value): int storing the global count of grads (total number of gradients taken by all workers).
                                    This value is updated by adding to it the "new_grads" (see "sync_process" doc) from every worker.
                                    This mp.Value is thus updated by world_size "listen_given_rank" processes, and used by the orchestrating process to kill all processes
                                    when the target number of grads is reached.
- lock (mp.Lock): multiprocessing lock to make sure that the nb_grad_tot_so_far is edited by only one process at a time, so that no "new gradients" are thrown out
                    by a multiprocessing bug.
- log (logger): to print messages in the logs if needed.
"""
function listen_given_rank(
    rank,
    world_size,
    queue,
    nb_tot_grad_so_far,
    lock,
    log,
)
    # initialize the process group for rank communications
    process_group = dist.init_process_group(
        backend = "gloo",
        init_method = "tcp://" +
                      os.environ["MASTER_ADDR"] +
                      ":" +
                      str(int(os.environ["MASTER_PORT"]) + 1),
        rank = rank + world_size,
        world_size = 2 * world_size + 1,
    )
    # initialize the placeholder tensor
    tensor_other_rank = torch.zeros(1)
    # while the master process does not send the order to kill all processes
    while tensor_other_rank.data != -2
        # receive information that worker rank is available for communications.
        # the act of receiving a message is the signal itself.
        # the inside of 'tensor_other_rank' variable contains the "new grads" performed by rank since last communication.
        dist.recv(tensor_other_rank, rank, process_group)
        # acquire the lock to edit the global count of grads
        lock.acquire()
        # add the new grads
        nb_tot_grad_so_far.value += int(tensor_other_rank.data)
        # signal the orchestrating process that worker nb rank is available for communication
        queue.put(rank)
        # release the lock so that another process can edit the variable
        lock.release()
    end
end


"""
### Orchestrating process hosted on worker 0.
This process accomplishes 2 things:
* Group available workers by pairs for p2p communication, according to the given graph topology, and trying to minimize latency
    by pairing together workers that were the first to be available to communicate.
* Signal to all processes when the target number of grads have been reached, so that computations & communication can stop.

Parameters:
- world_size (int): the total number of workers.
- nb_grad_tot_goal (int): The target number of total nb of grads performed by all workers.
                            When it is reached, this process sends the signal to all other to stop all computations & communications.
- log (logger): to print messages in the logs if needed.
- graph_topology (str): Graph topology to use to make p2p communication (dictates which edges can be used).
                        Currently supports either of ['complete'].
- deterministic_neighbor (bool): whether or not to schedule the p2p communications.
                                    if True, if at the next step, worker i is supposed to communicate with j,
                                    i will wait for j to be available to communicate.
                                    if False, i will communicate faster, by just picking one of its available neighbor.
"""
function master_process(
    world_size,
    nb_grad_tot_goal,
    log,
    graph_topology,
    deterministic_neighbor,
)
    # Initialize multiprocessing variables shared with the "listen_given_rank" processes, also hosted by worker 0.
    # Queue containing the rank of available workers. 
    # Ranks are enqueued by their corresponding "listen_given_rank" process, and dequeued here to make pairs.
    queue = mp.Queue()
    # lock to protect the edit of variables shared by those multiple processes.
    lock = mp.Lock()
    # Init an int storing the global count of grads (total number of gradients taken by all workers).
    # this mp.Value is shared with all "listen_given_rank" processes, so that they can edit it.
    # When the target number of grads is reached, this process sends the signal to all other to stop all computations & communications.
    nb_tot_grad_so_far = mp.Value("i", 0)
    list_processes = []
    # launch the listening processes for each rank
    for rank in range(world_size)
        listen_process = Process(
            target = listen_given_rank,
            args = (rank, world_size, queue, nb_tot_grad_so_far, lock, log),
        )
        listen_process.start()
        list_processes.append(listen_process)
    end
    # initialize the process group for rank communications
    process_group = dist.init_process_group(
        backend = "gloo",
        init_method = "tcp://" +
                      os.environ["MASTER_ADDR"] +
                      ":" +
                      str(int(os.environ["MASTER_PORT"]) + 1),
        rank = 2 * world_size,
        world_size = 2 * world_size + 1,
    )
    # tuple of ranks stores the first 2 available workers to communicate
    # only used if the graph_topology is not complete.
    tuple_of_ranks = []
    # init placeholder tensors to communicate with the appropriate "sync_process" process.
    tensor_rank_0 = torch.zeros(1)
    tensor_rank_1 = torch.zeros(1)

    # if the topology is not complete
    if graph_topology != "complete"
        throw(ValueError("Supported graph topologies are ['complete']."))
    end

    # while the total number of grad is not reached
    while nb_tot_grad_so_far.value < nb_grad_tot_goal
        # get the rank of the first available worker
        tuple_of_ranks.append(queue.get())
        # if 2 workers are available for communication
        if len(tuple_of_ranks) == 2
            # gather their ranks
            tensor_rank_0[0] = tuple_of_ranks[0]
            tensor_rank_1[0] = tuple_of_ranks[1]
            # send their ranks to each other
            dist.send(tensor_rank_0, tuple_of_ranks[1], process_group)
            dist.send(tensor_rank_1, tuple_of_ranks[0], process_group)
            # re-initialize the tuple as an empty one
            tuple_of_ranks = []
        end
    end
    # when we go out of the while loop, send to everybody the message to stop processes
    kill_process_signal = torch.ones(1) * (-2)
    for rank in range(world_size)
        dist.send(kill_process_signal, rank, process_group)
    end

    # terminates all processes
    for p in list_processes
        p.join()
    end
end

