var documenterSearchIndex = {"docs":
[{"location":"index.html#ACiD.jl","page":"Home","title":"ACiD.jl","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"This is an implementation of an asynchronous multiprocessing optimization algorithm with a continuous local momentum called A²CiD² in the Julia programming language as introduced in [[1]].","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"[!NOTE] There is also an official demo by the original authors of [[1]] in Python: AdelNabli/ACiD","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"<!– ","category":"page"},{"location":"index.html#Usage","page":"Home","title":"Usage","text":"","category":"section"},{"location":"index.html#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"index.html#Example","page":"Home","title":"Example","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"s = \"Julia syntax highlighting\";\nprintln(s);","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"–>","category":"page"},{"location":"index.html#Functions-Index","page":"Home","title":"Functions Index","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"acid_ode\ngossip_process\nmaster_process\nsync_process\ndo_send\ndo_recv\nlisten_given_rank","category":"page"},{"location":"index.html#ACiD.acid_ode","page":"Home","title":"ACiD.acid_ode","text":"Integrate the ODE for the continuous momentum, see https://arxiv.org/pdf/2306.08289.pdf for details.\n\nUpdate parameters (paramscom and paramscom_tilde) in-place.\n\nParameters:\n\nparams_com (torch.tensor): 1D tensor containing all of the models learnable parameters.\nparamscomtilde (torch.tensor): \"momentum\" variable, same size as paramscom, mixing with paramscom to obtain acceleration.\nodematrix (torch.tensor): a 2x2 matrix storing the parameters of the linear mixing between params and paramstilde.\nt_old (float): time of the last local update.\nt_new (float): time of the current update.\ndeltatgrad (float): time that it takes to compute a grad step. Used to re-normalize time, as done in the paper.\n\n\n\n\n\n","category":"function"},{"location":"index.html#ACiD.gossip_process","page":"Home","title":"ACiD.gossip_process","text":"Gossip routine for the p2p averaging of the model's parameters running in the background.\n\nAverage the parameters of all the workers at the beginning (to start from a common initialization), and at the end.\nUse the mp.Variable \"rank_other\" to communicate with the orchestring process   that pairs available workers together to perform p2p communications, allowing this   function to know with which rank to communicate next.\nDepending on deterministic_com, implement or not a P.P.P for the communication process:   if true, a random number of p2p communications between 2 grad steps are done, following a poisson law.\nWhen the orchestrating process counted that the right number of grad step have been performed in total,   signal it back to this process (stops the communication routine), which signals to the main process to stop performing grad steps.\n\nParameters:\n\nrank (int): our rank id in the distributed setting.\nlocal_rank (int): the local rank of the worker inside its compute node (to create a Cuda Stream in the right GPU).\nworld_size (int): the total number of workers.\nrankother (mp.Value): a multiprocessing Value to store the id of the rank of the next communication. It is updated                           by the orchestrating process pairing workers together, and re-initialized by this one after a communication.                           if rankother.value == -1: (base value) no peer has been found yet.                           if rankother.value == -2: signal from the orchestrating process that enough gradients have been computed in total,                                                   stops the communication process.                           if rankother.value not in [-1, -2]: contains the rank of the worker we are supposed to communicate with next.\nparams_com (torch.tensor): 1D tensor containing the model's parameters.\nparamsother (torch.tensor): 1D tensor, placeholder to receive the paramscom of the worker with whom we communicate.\nbarriersyncaveraging (mp.Barrier): a barrier used to communicate with the synchronization process.                                       When we meet this barrier, we signal to the sync process that we finished our previous communication,                                       and are available for the next one, so that it can begin to look for another available peer to connect                                       to for the next p2p communication.\ncontinuegradroutine (mp.Value containing a bool): whether or not the grad process should continue.                                                       Initialized at 1 (true). Is put to 0 (False) when the orchestrating                                                       process signals to us that the total number of gradients quota has been met.\nbarrierendinit (mp.Barrier): a barrier to signal to the init function of ADP's class that the initializing average of the parameters                                   has been performed, and that ADP can resume its init.\nbarriercomgrad (mp.Barrier): a barrier to make sure a certain amount of communication has been made between 2 grads.                                   Also used to make sure a certain amount of grad have been performed between 2 comm if rate_com < 1.\nlog (logger): to print messages in the logs if needed.\ncomhistory (list of mp.Value): list of size worldsize. Used to logg how many times this worker communicated with each of its peers.\ncountcomslocal (mp.Value): a count of the number of p2p communications this worker has done.\nrate_com (float): the rate at which p2p communications are done (in expectation) compared to local grad steps.\napply_acid (bool): whether or not to apply ACiD momentum. If True, the communication is an \"event\" triggering a momentum update.\nparamscomtilde (torch.tensor): \"momentum\" variable, same size as paramscom, mixing with paramscom to obtain acceleration.\nodematrix (torch.tensor): a 2x2 matrix storing the parameters of the linear mixing between params and paramstilde.\ntlastspike (float): time of the last local update to params_com (be it a communication or gradient one).\ndeltatgrad (mp.Value storing a double): the variable keeping track of the time that it takes to make a grad step.\nbetatilde (float): the \u0007lphatilde value to use in ACiD.\ndeterministic_com (bool): whether or not to schedule to use Poisson Point Processes for the communications.                           if True, a random number of p2p communications between 2 grad steps are done, following a poisson law.\n\n\n\n\n\n","category":"function"},{"location":"index.html#ACiD.master_process","page":"Home","title":"ACiD.master_process","text":"Orchestrating process hosted on worker 0.\n\nThis process accomplishes 2 things:\n\nGroup available workers by pairs for p2p communication, according to the given graph topology, and trying to minimize latency   by pairing together workers that were the first to be available to communicate.\nSignal to all processes when the target number of grads have been reached, so that computations & communication can stop.\n\nParameters:\n\nworld_size (int): the total number of workers.\nnbgradtot_goal (int): The target number of total nb of grads performed by all workers.                           When it is reached, this process sends the signal to all other to stop all computations & communications.\nlog (logger): to print messages in the logs if needed.\ngraph_topology (str): Graph topology to use to make p2p communication (dictates which edges can be used).                       Currently supports either of ['complete'].\ndeterministic_neighbor (bool): whether or not to schedule the p2p communications.                                   if True, if at the next step, worker i is supposed to communicate with j,                                   i will wait for j to be available to communicate.                                   if False, i will communicate faster, by just picking one of its available neighbor.\n\n\n\n\n\n","category":"function"},{"location":"index.html#ACiD.sync_process","page":"Home","title":"ACiD.sync_process","text":"Process run by every worker in the background.\n\nThis process allows each worker to communicate with the \"orchestrating\" master process (hosted by worker 0). The goal is to signal to the master process when this worker is available for communication, and to gather from the master process the rank of the peer with which we are supposed to communicate. When received, this information \"rankother\" is sent to the p2paveraging process run in parallel in this worker, so that the p2p averaging process knows with which worker to communicate next. This process will communicate with one of the worldsize \"listengivenrank\" processes hosted at worker 0, which has worldsize + 1 processes run in parallel:     * one to \"listento\" each one of the syncprocess run by each worker.     * one \"orchestrating\" process, dedicated to make pairs of workers. So, in total, there are 2*worldsize + 1 processes that need to communicate with each other (only sending ints), so initialize a processgroup using gloo backend here.\n\nParameters:\n\nrank (int): our rank id in the distributed setting.\nworld_size (int): the total number of workers.\nrankother (mp.Value): a multiprocessing Value to store the id of the rank of the next communication.                           It is updated here, based on the information given by the master process, to signal to the p2paveraging process                           run in parallel in this worker which peer to communicate wiith next.                           if rank_other.value == -2: signal from the orchestrating process that enough gradients have been computed in total,                                                   stops the communication process.\nnew_grads (mp.Value): a multiprocessing Value updated by the process and the main one, counting how many new grad steps have been performed                       by this worker since last communication. This is used by the master process to count the total number of grad done,                       and initiate the \"kill\" of all processes when the right amount of grad steps have been performed in total.\nbarriersyncaveraging (mp.Barrier): a barrier used to communicate with the p2p_averaging process.                                       When the averaging process meets this barrier, it signals to this process that the worker                                       is available for the next communication, so we can begin to look for another available peer to connect                                       to by sending our rank information to the master process which will realize the pairing.\nlog (logger): to print messages in the logs if needed.\n\n\n\n\n\n","category":"function"},{"location":"index.html#ACiD.do_send","page":"Home","title":"ACiD.do_send","text":"The send THEN receive function.\n\nExpects that the peer with whom we communicate runs the symetric function receive THEN send. The p2p communication edits in-place the values of the parameters paramscom and paramscomtilde (if applyacid).\n\nParameters:\n\nparams_com (torch.tensor): 1D tensor containing the model's parameters.\nparamsotherworker (torch.tensor): 1D tensor, placeholder to receive the params_com of the worker with whom we communicate.\nprocessgroup (a torch distributed processgroup): specifies the process_group to use for the p2p communications.\nother_rank (int): the rank of the worker we communicate with.\napply_acid (bool): whether or not to apply ACiD momentum. If true, the communication is an \"event\" triggering a momentum update.\nparamscomtilde (torch.tensor): \"momentum\" variable, same size as paramscom, mixing with paramscom to obtain acceleration.\nodematrix (torch.tensor): a 2x2 matrix storing the parameters of the linear mixing between params and paramstilde.\ntlastspike (float): time of the last local update to params_com (be it a communication or gradient one).\ndeltatgrad (mp.Value storing a double): the variable keeping track of the time that it takes to make a grad step.\nbetatilde (float): the \u0007lphatilde value to use in ACiD.\n\n\n\n\n\n","category":"function"},{"location":"index.html#ACiD.do_recv","page":"Home","title":"ACiD.do_recv","text":"The receive THEN send function.\n\nExpects that the peer with whom we communicate runs the symetric function send THEN receive. The p2p communication edits in-place the values of the parameters paramscom and paramscomtilde (if applyacid).\n\nParameters:\n\nparams_com (torch.tensor): 1D tensor containing the model's parameters.\nparamsotherworker (torch.tensor): 1D tensor, placeholder to receive the params_com of the worker with whom we communicate.\nprocessgroup (a torch distributed processgroup): specifies the process_group to use for the p2p communications.\nother_rank (int): the rank of the worker we communicate with.\napply_acid (bool): whether or not to apply ACiD momentum. If true, the communication is an \"event\" triggering a momentum update.\nparamscomtilde (torch.tensor): \"momentum\" variable, same size as paramscom, mixing with paramscom to obtain acceleration.\nodematrix (torch.tensor): a 2x2 matrix storing the parameters of the linear mixing between params and paramstilde.\ntlastspike (float): time of the last local update to params_com (be it a communication or gradient one).\ndeltatgrad (mp.Value storing a double): the variable keeping track of the time that it takes to make a grad step.\nbetatilde (float): the \u0007lphatilde value to use in ACiD.\n\n\n\n\n\n","category":"function"},{"location":"index.html#ACiD.listen_given_rank","page":"Home","title":"ACiD.listen_given_rank","text":"Process run in the background of worker 0.\n\nIts goal is to listen to one specific worker (specifically, its \"syncprocess\" process), and to send it information coming from the orchestrating process also hosted by worker 0. The main goal of this function is to put to the mp.Queue the rank of the worker it is listening to when this worker sent, through its \"syncprocess\" function, the signal that its corresponding worker was available for a communication. Then, as this mp.Queue is shared with the orchestrating process, the orchestrating process can receive the information and pair the worker with another one.\n\nParameters:\n\nrank (int): our rank id in the distributed setting.\nworld_size (int): the total number of workers.\nqueue (mp.Queue): queue containing the ranks of all available workers for communication.                   The orchestrating process then only needs to \"de-queue\" the ranks to make pairs, insuring that the communications are performed in FIFO style,                   minimizing latency.\nnbgradtotsofar (mp.Value): int storing the global count of grads (total number of gradients taken by all workers).                                   This value is updated by adding to it the \"newgrads\" (see \"syncprocess\" doc) from every worker.                                   This mp.Value is thus updated by worldsize \"listengiven_rank\" processes, and used by the orchestrating process to kill all processes                                   when the target number of grads is reached.\nlock (mp.Lock): multiprocessing lock to make sure that the nbgradtotsofar is edited by only one process at a time, so that no \"new gradients\" are thrown out                   by a multiprocessing bug.\nlog (logger): to print messages in the logs if needed.\n\n\n\n\n\n","category":"function"},{"location":"index.html#References","page":"Home","title":"References","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"[1] <span id=\"[1]\">A. Nabli, E. Belilovsky, and E. Oyallon, “A²CiD²: Accelerating Asynchronous Communication in Decentralized Deep Learning,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: (Image: arXiv) (Image: DOI:10.48550/ARXIV.2306.08289)</span>","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"[1]: #[1] \"A. Nabli, E. Belilovsky, and E. Oyallon, “A²CiD²: Accelerating Asynchronous Communication in Decentralized Deep Learning,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023.\"","category":"page"}]
}