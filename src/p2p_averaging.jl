# import time
# import torch
# import numpy as np
# import torch.distributed as dist
# from utils.acid_utils import acid_ode

using ExponentialUtilities

# for the purpose of testing the documentation
export acid_ode, do_send, do_recv, gossip_process

"""
### Integrate the ODE for the continuous momentum, see https://arxiv.org/pdf/2306.08289.pdf for details.
Update parameters (params_com and params_com_tilde) in-place.

Parameters:
- params_com (torch.tensor): 1D tensor containing all of the models learnable parameters.
- params_com_tilde (torch.tensor): "momentum" variable, same size as params_com, mixing with params_com to obtain acceleration.
- ode_matrix (torch.tensor): a 2x2 matrix storing the parameters of the linear mixing between params and params_tilde.
- t_old (float): time of the last local update.
- t_new (float): time of the current update.
- delta_t_grad (float): time that it takes to compute a grad step. Used to re-normalize time, as done in the paper.
"""
function acid_ode(
    params_com::Vector,
    params_com_tilde::Vector,
    ode_matrix::Matrix,
    t_old::Real,
    t_new::Real,
    delta_t_grad::Real,
)
    # Compute the exponential of the matrix of the ode system
    # between t_old and t_new (we re-normalize time using delta_t_grad as the unit of time)
    exp_M = exponential!(ode_matrix * (t_new - t_old) / delta_t_grad)
    a, b, c, d = exp_M[1, 1], exp_M[1, 2], exp_M[2, 1], exp_M[2, 2]
    # Do the mixing in-place, so first remembers the value of params
    params_old = params_com.detach().clone()
    # matrix multiplication
    params_com.mul_(a).add_(params_com_tilde, alpha = b)
    params_com_tilde.mul_(d).add_(params_old, alpha = c)
end

"""
### The send THEN receive function.
Expects that the peer with whom we communicate runs the symetric function receive THEN send.
The p2p communication edits in-place the values of the parameters params_com and params_com_tilde (if apply_acid).

Parameters:
- params_com (torch.tensor): 1D tensor containing the model's parameters.
- params_other_worker (torch.tensor): 1D tensor, placeholder to receive the params_com of the worker with whom we communicate.
- process_group (a torch distributed process_group): specifies the process_group to use for the p2p communications.
- other_rank (int): the rank of the worker we communicate with.
- apply_acid (bool): whether or not to apply ACiD momentum. If true, the communication is an "event" triggering a momentum update.
- params_com_tilde (torch.tensor): "momentum" variable, same size as params_com, mixing with params_com to obtain acceleration.
- ode_matrix (torch.tensor): a 2x2 matrix storing the parameters of the linear mixing between params and params_tilde.
- t_last_spike (float): time of the last local update to params_com (be it a communication or gradient one).
- delta_t_grad (mp.Value storing a double): the variable keeping track of the time that it takes to make a grad step.
- beta_tilde (float): the \alpha_tilde value to use in ACiD.
"""
function do_send(
    params_com,
    params_other_worker,
    process_group,
    other_rank,
    apply_acid,
    params_com_tilde,
    ode_matrix,
    t_last_spike,
    delta_t_grad,
    beta_tilde,
)


    # sends and receives the params to and from an other worker
    dist.send(params_com, other_rank, process_group)
    dist.recv(params_other_worker, other_rank, process_group)
    if apply_acid
        # retrieve the times
        t_old = t_last_spike.value
        t_new = time.time()
        # apply continuous momentum
        acid_ode(
            params_com,
            params_com_tilde,
            ode_matrix,
            t_old,
            t_new,
            delta_t_grad.value,
        )
        # update the t spike var
        t_last_spike.value = t_new
        # update params_com_tilde
        params_com_tilde.add_(beta_tilde * (params_other_worker - params_com))
        # inplace average of parameters
        params_com.lerp_(params_other_worker, 0.5)
    end
end


"""
### The receive THEN send function.
Expects that the peer with whom we communicate runs the symetric function send THEN receive.
The p2p communication edits in-place the values of the parameters params_com and params_com_tilde (if apply_acid).

Parameters:
- params_com (torch.tensor): 1D tensor containing the model's parameters.
- params_other_worker (torch.tensor): 1D tensor, placeholder to receive the params_com of the worker with whom we communicate.
- process_group (a torch distributed process_group): specifies the process_group to use for the p2p communications.
- other_rank (int): the rank of the worker we communicate with.
- apply_acid (bool): whether or not to apply ACiD momentum. If true, the communication is an "event" triggering a momentum update.
- params_com_tilde (torch.tensor): "momentum" variable, same size as params_com, mixing with params_com to obtain acceleration.
- ode_matrix (torch.tensor): a 2x2 matrix storing the parameters of the linear mixing between params and params_tilde.
- t_last_spike (float): time of the last local update to params_com (be it a communication or gradient one).
- delta_t_grad (mp.Value storing a double): the variable keeping track of the time that it takes to make a grad step.
- beta_tilde (float): the \alpha_tilde value to use in ACiD.
"""
function do_recv(
    params_com,
    params_other_worker,
    process_group,
    other_rank,
    apply_acid,
    params_com_tilde,
    ode_matrix,
    t_last_spike,
    delta_t_grad,
    beta_tilde,
)


    # receives and sends the params to and from an other worker
    dist.recv(params_other_worker, other_rank, process_group)
    dist.send(params_com, other_rank, process_group)
    if apply_acid
        # retrieve the times
        t_old = t_last_spike.value
        t_new = time.time()
        # apply continuous momentum
        acid_ode(
            params_com,
            params_com_tilde,
            ode_matrix,
            t_old,
            t_new,
            delta_t_grad.value,
        )
        # update the t spike var
        t_last_spike.value = t_new
        # update params_com_tilde
        params_com_tilde.add_(beta_tilde * (params_other_worker - params_com))
    end
    # inplace average of parameters
    params_com.lerp_(params_other_worker, 0.5)
end



"""
### Gossip routine for the p2p averaging of the model's parameters running in the background.

* Average the parameters of all the workers at the beginning (to start from a common initialization), and at the end.
* Use the mp.Variable "rank_other" to communicate with the orchestring process
    that pairs available workers together to perform p2p communications, allowing this
    function to know with which rank to communicate next.
* Depending on deterministic_com, implement or not a P.P.P for the communication process:
    if true, a random number of p2p communications between 2 grad steps are done, following a poisson law.
* When the orchestrating process counted that the right number of grad step have been performed in total,
    signal it back to this process (stops the communication routine), which signals to the main process to stop performing grad steps.

Parameters:
- rank (int): our rank id in the distributed setting.
- local_rank (int): the local rank of the worker inside its compute node (to create a Cuda Stream in the right GPU).
- world_size (int): the total number of workers.
- rank_other (mp.Value): a multiprocessing Value to store the id of the rank of the next communication. It is updated
                            by the orchestrating process pairing workers together, and re-initialized by this one after a communication.
                            if rank_other.value == -1: (base value) no peer has been found yet.
                            if rank_other.value == -2: signal from the orchestrating process that enough gradients have been computed in total,
                                                    stops the communication process.
                            if rank_other.value not in [-1, -2]: contains the rank of the worker we are supposed to communicate with next.
- params_com (torch.tensor): 1D tensor containing the model's parameters.
- params_other (torch.tensor): 1D tensor, placeholder to receive the params_com of the worker with whom we communicate.
- barrier_sync_averaging (mp.Barrier): a barrier used to communicate with the synchronization process.
                                        When we meet this barrier, we signal to the sync process that we finished our previous communication,
                                        and are available for the next one, so that it can begin to look for another available peer to connect
                                        to for the next p2p communication.
- continue_grad_routine (mp.Value containing a bool): whether or not the grad process should continue.
                                                        Initialized at 1 (true). Is put to 0 (False) when the orchestrating
                                                        process signals to us that the total number of gradients quota has been met.
- barrier_end_init (mp.Barrier): a barrier to signal to the __init__ function of ADP's class that the initializing average of the parameters
                                    has been performed, and that ADP can resume its init.
- barrier_com_grad (mp.Barrier): a barrier to make sure a certain amount of communication has been made between 2 grads.
                                    Also used to make sure a certain amount of grad have been performed between 2 comm if rate_com < 1.
- log (logger): to print messages in the logs if needed.
- com_history (list of mp.Value): list of size world_size. Used to logg how many times this worker communicated with each of its peers.
- count_coms_local (mp.Value): a count of the number of p2p communications this worker has done.
- rate_com (float): the rate at which p2p communications are done (in expectation) compared to local grad steps.
- apply_acid (bool): whether or not to apply ACiD momentum. If True, the communication is an "event" triggering a momentum update.
- params_com_tilde (torch.tensor): "momentum" variable, same size as params_com, mixing with params_com to obtain acceleration.
- ode_matrix (torch.tensor): a 2x2 matrix storing the parameters of the linear mixing between params and params_tilde.
- t_last_spike (float): time of the last local update to params_com (be it a communication or gradient one).
- delta_t_grad (mp.Value storing a double): the variable keeping track of the time that it takes to make a grad step.
- beta_tilde (float): the \alpha_tilde value to use in ACiD.
- deterministic_com (bool): whether or not to schedule to use Poisson Point Processes for the communications.
                            if True, a random number of p2p communications between 2 grad steps are done, following a poisson law.

"""
function gossip_process(
    rank,
    local_rank,
    world_size,
    rank_other,
    params_com,
    params_com_other,
    barrier_sync_averaging,
    continue_grad_routine,
    barrier_end_init,
    barrier_com_grad,
    log,
    com_history,
    count_coms_local,
    rate_com,
    apply_acid,
    params_com_tilde,
    ode_matrix,
    t_last_spike,
    delta_t_grad,
    beta_tilde,
    deterministic_com,
)

    # initialize the process group for communications
    process_group = dist.init_process_group(
        backend = "nccl",
        rank = rank,
        world_size = world_size,
    )
    # initialize model weights by performing a first all-reduce
    torch.cuda.synchronize()
    dist.all_reduce(params_com, group = process_group, op = dist.ReduceOp.SUM)
    params_com.mul_(1 / world_size)
    # initialize the right momentum variable
    if apply_acid
        # initialize the momentum variable
        params_com_tilde.copy_(params_com)
    end
    # signal the end of the initialization to the main process
    barrier_end_init.wait()
    # create the gossip stream
    gossip_stream = torch.cuda.Stream(device = local_rank)
    count_coms_next_wait = 1

    # we do everything in the gossip stream
    # with torch.cuda.stream(gossip_stream)
    while true
        rank_other_here = rank_other.value
        # wait the rank of an other available worker
        while rank_other_here == -1
            rank_other_here = rank_other.value
        end
        # rank_other is equal to -2 when we made enough grad steps in total
        # so there is no need to communicate anymore
        if rank_other_here == -2
            barrier_sync_averaging.abort()
            break
        end
        # averaging with rank_other.
        # the order in which to perform the send and receive operations is dictated by the test rank_other < rank.
        if rank_other_here < rank
            do_send(
                params_com,
                params_com_other,
                process_group,
                rank_other_here,
                apply_acid,
                params_com_tilde,
                ode_matrix,
                t_last_spike,
                delta_t_grad,
                beta_tilde,
            )
        else
            do_recv(
                params_com,
                params_com_other,
                process_group,
                rank_other_here,
                apply_acid,
                params_com_tilde,
                ode_matrix,
                t_last_spike,
                delta_t_grad,
                beta_tilde,
            )
        end
        # logs the communication
        count_coms_local.value += 1
        count_com_rank = com_history[rank_other_here]
        count_com_rank.value += 1
        # wait or synchronize with the grad process
        if rate_com >= 1
            if count_coms_local.value >= count_coms_next_wait
                # Wait for 1 averaging step before grad
                barrier_com_grad.wait()
                barrier_com_grad.reset()
                # if coms are deterministic
                if deterministic_com
                    # add the precise amount of com before the next grad step
                    count_coms_next_wait += rate_com
                else
                    # else, uses poisson law to implement the Poisson Point Processes for communications
                    count_coms_next_wait +=
                        np.random.poisson(lam = rate_com, size = None)
                end
            end
        else
            barrier_com_grad.wait()
        end
        # re-initialize the mp.Value var for next round
        rank_other.value = -1
        # signal to the synchronization process we are available for communication
        try
            barrier_sync_averaging.wait()
            t_beg_com = time.time()
        catch
            # only way this fails is barrier already broken by sync process
            break
        end
    end
    # signal to grad process to stop
    continue_grad_routine.value = 0
    try
        barrier_com_grad.abort()
    catch
        nothing
    end
    # alll reduce the params at the end of the training
    dist.barrier(group = process_group)
    torch.cuda.synchronize()
    dist.all_reduce(params_com, group = process_group, op = dist.ReduceOp.SUM)
    params_com.mul_(1 / world_size)
end