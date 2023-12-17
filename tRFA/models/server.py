import numpy as np
from scipy.stats import norm
import random
import torch
import copy
from torch.multiprocessing import Pool
from baseline_constants import *
from pgaAtcUtils import *
# MEBwO
import sys
sys.path.append('../../MEBwO/src')
from meb.ball import MEBwO

# from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY, AVG_LOSS_KEY
# from baseline_constants import CORRUPTION_OMNISCIENT_KEY, CORRUPTION_GAUSS_KEY, CORRUPTION_GAUSS2_KEY, \
from baseline_constants import *
MAX_UPDATE_NORM, CORRUPTION_EMPIRE_KEY, CORRUPTION_LITTLE_KEY, CORRUPTION_SIGNFLIP_KEY, CORRUPTION_PGA_KEY


class Server:

    def __init__(self, model):
        self.model = model  # global model of the server.
        self.selected_clients = []
        self.updates = []
        self.rng = model.rng  # use random number generator of the model
        self.total_num_comm_rounds = 0
        self.eta = None

    def select_clients(self, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        self.selected_clients = self.rng.sample(possible_clients, num_clients)

        return [(len(c.train_data['y']), len(c.eval_data['y'])) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None,
                    clients=None, lr=None, lmbda=None):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
            lr: learning rate to use
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """

        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        losses = []

        chosen_clients = clients

        for c in chosen_clients:
            self.model.send_to([c])  # reset client model
            sys_metrics[c.id][BYTES_READ_KEY] += self.model.size
            if lmbda is not None:
                c._model.optimizer.lmbda = lmbda
            if lr is not None:
                c._model.optimizer.learning_rate = lr

            comp, num_samples, averaged_loss, update = c.train(num_epochs, batch_size, minibatch, lr)
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
            losses.append(averaged_loss)

            self.updates.append((num_samples, update))
            # print("server.py line85 num_samples: ", num_samples)
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += self.model.size
            # sys_metrics[c.id][AVG_LOSS_KEY] = averaged_loss
        
        # with Pool() as pool:
        #     def worker(c):
        #         self.model.send_to([c])  # reset client model
        #         sys_metrics[c.id][BYTES_READ_KEY] += self.model.size
        #         if lmbda is not None:
        #             c._model.optimizer.lmbda = lmbda
        #         if lr is not None:
        #             c._model.optimizer.learning_rate = lr

        #         comp, num_samples, averaged_loss, update = c.train(num_epochs, batch_size, minibatch, lr)
        #         sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
        #         losses.append(averaged_loss)

        #         self.updates.append((num_samples, update))
        #         # print("server.py line85 num_samples: ", num_samples)
        #         sys_metrics[c.id][BYTES_WRITTEN_KEY] += self.model.size
            
        #     pool.map(worker, chosen_clients)
        #     pool.close()
        #     pool.join()

        avg_loss = np.nan if len(losses) == 0 else \
            np.average(losses, weights=[len(c.train_data['y']) for c in chosen_clients])
        return sys_metrics, avg_loss, losses

    def update_model(
        self, aggregation, corruption=None, corrupted_client_ids=frozenset(), maxiter=4,
        fraction_to_discard=0, norm_bound=None,corp_client_rate = None, mebwo_alg='shenmaier',little_z=None,dev_type=None
    ):
        """
        dev_type: param for pga_attack_mebra only
        """
        is_corrupted = [(client.id in corrupted_client_ids) for client in self.selected_clients]
        if corruption == CORRUPTION_OMNISCIENT_KEY and any(is_corrupted):
            # compute omniscient update
            avg = self.model.weighted_average_oracle([u[1] for u in self.updates], [u[0] for u in self.updates])
            num_pts = sum([u[0] for u in self.updates])

            corrupted_updates = [u for c, u in zip(is_corrupted, self.updates) if c]
            corrupted_avg = self.model.weighted_average_oracle([u[1] for u in corrupted_updates],
                                                               [u[0] for u in corrupted_updates])
            num_corrupt_pts = sum([u[0] for u in corrupted_updates])
            omniscient_update = corrupted_avg - 2 * num_pts / num_corrupt_pts * avg
            # omniscient_update = [wc - 2 * num_pts / num_corrupt_pts * w_avg for wc, w_avg in zip(corrupted_avg, avg)]
            # change self.updates to reflect omniscient update
            for i in range(len(self.updates)):
                if is_corrupted[i]:
                    self.updates[i] = (self.updates[i][0], omniscient_update)
        # gauss attack1: attack all updates
        elif corruption == CORRUPTION_GAUSS_KEY and any(is_corrupted):
            for i in range(len(self.updates)):
                if is_corrupted[i]:
                    u = self.updates[i][1]
                    u1 = np.random.randn(*u.shape) * np.std(u)
                    self.updates[i] = (self.updates[i][0], u1)
        # gauss attack2: attack corrupted clients' updates
        elif corruption == CORRUPTION_GAUSS2_KEY and any(is_corrupted):
            # self.model.model = sent140.erm_log_reg.ClientModel
            w = self.model.model.optimizer.w
            for i in range(len(self.updates)):
                if is_corrupted[i]:
                    u = self.updates[i][1]
                    u1 = np.random.randn(*u.shape)
                    u1 *= np.linalg.norm(w) / np.linalg.norm(u1)  # apply norm = model 
                    u1 = u1 - w  # difference to w
                    self.updates[i] = (self.updates[i][0], u1)
        # empire attack --paper: fall of empire
        elif corruption == CORRUPTION_EMPIRE_KEY and any(is_corrupted):
            # debug ok
            # avg = self.model.weighted_average_oracle([u[1] for u in self.updates], [u[0] for u in self.updates])
            # print("server.py line133 all workers' avg[0:5]: ", avg[0:5])
            # honest workers' weighted average update
            hon_avg = self.model.weighted_average_oracle([u[1] for i,u in enumerate(self.updates) if not is_corrupted[i]], 
            [u[0] for i,u in enumerate(self.updates) if not is_corrupted[i]])
            # print("server.py line133 honest workers' avg[0:5]: ", hon_avg[0:5])
            corpu = -0.1*hon_avg
            # print("server.py line133 corpu = -0.1*hon_avg: ", corpu)
            for i in range(len(self.updates)):
                if is_corrupted[i]:
                    self.updates[i] = (self.updates[i][0], corpu)
        # little attack --paper: a little is enough
        elif corruption == CORRUPTION_LITTLE_KEY and any(is_corrupted):
            # n = len(self.selected_clients)
            # m = sum(is_corrupted)
            # s = int(n / 2 + 1) - m
            # p = (n - m - s) / (n - m)
            # z = norm.ppf(p)
            #z = 100 # for sent140
            #z = 100 #for femnist
            # z = 20 #for femnist_cnn
            z=little_z
            # debug ok
            print("In server.py line180 little attack, z: ", z) # "n: ", n, " m: ", m, " s: ", s, " p: ", p,

            grads_mean =self.model.weighted_average_oracle([u[1] for i,u in enumerate(self.updates) if not is_corrupted[i]], 
            [u[0] for i,u in enumerate(self.updates) if not is_corrupted[i]])
            # print("server.py line155 grads_mean: ", grads_mean)
            num_pts = sum([u[0] for u in self.updates])
            grads_stdev = np.var([ u[1]*u[0]/num_pts for i,u in enumerate(self.updates) if not is_corrupted[i]], axis=0) ** 0.5
            # print("server.py line155 grads_stdev: ", grads_stdev)
            grads_mean[:] -= z * grads_stdev[:]
            # print("server.py line155 final grads_mean: ", grads_mean)
            for i in range(len(self.updates)):
                if is_corrupted[i]:
                    self.updates[i] = (self.updates[i][0], grads_mean)
        # sign-flipping attack: every Byzantine worker sends the negative of its update to the server.
        elif corruption == CORRUPTION_SIGNFLIP_KEY and any(is_corrupted):
            for i in range(len(self.updates)):
                if is_corrupted[i]:
                    # print("server.py line168 row updata: ", self.updates[i][1])
                    self.updates[i] = (self.updates[i][0], -self.updates[i][1])
                    # print("server.py line168 final updata: ", self.updates[i][1])
        elif corruption == CORRUPTION_PGA_KEY:
            # Paper: Back to the Drawing Board: A Critical Evaluation of Poisoning Attacks on Production Federated Learning
            updates = [u[1] for u in self.updates]  # updates of all clients, both benign and malicious
            # print("in server.py line210: updates[:5]: ",updates[:5])
            weights = [u[0] for u in self.updates]  # weights of all clients, both benign and malicious
            mean_up =self.model.weighted_average_oracle([u[1] for i,u in enumerate(self.updates) if not is_corrupted[i]], 
            [u[0] for i,u in enumerate(self.updates) if not is_corrupted[i]])

            if aggregation == AGGR_KRUM:
                mal_update = pga_attack_mkrum(updates=updates, weights=weights, mean_up=mean_up, is_corrupted=is_corrupted, fraction_to_discard=fraction_to_discard)
            elif aggregation == AGGR_MEBWO:
                mal_update = pga_attack_mebra(updates=updates, weights=weights, mean_up=mean_up, is_corrupted=is_corrupted, mebwo_alg=mebwo_alg, dev_type=dev_type)
            elif aggregation == AGGR_TRIM_MEAN:
                mal_update = pga_attack_trmean(updates=updates, weights=weights, mean_up=mean_up, is_corrupted=is_corrupted, fraction_to_discard=fraction_to_discard)
            elif aggregation == AGGR_MEAN:
                scale_factor = 1e20
                mal_update = (mean_up - 20 * np.sign(mean_up)) * scale_factor
            elif aggregation == AGGR_NORM_MEAN:
                mal_update = pga_attack_norm(updates=updates, weights=weights, mean_up=mean_up, is_corrupted=is_corrupted, norm_bound=norm_bound)
            elif aggregation == AGGR_GEO_MED:
                mal_update = pga_attack_gm(updates=updates, weights=weights, mean_up=mean_up, is_corrupted=is_corrupted, maxiter = maxiter)
            elif aggregation == AGGR_CO_MED:
                mal_update = pga_attack_median(updates=updates, weights=weights, mean_up=mean_up, is_corrupted=is_corrupted)
            else:
                raise ValueError('unimplemented pga attack for aggr : {}'.format(aggregation))
            
            for i in range(len(self.updates)):
                if is_corrupted[i]:
                    self.updates[i] = (self.updates[i][0], mal_update)

        elif corruption == CORRUPTION_P_X_KEY or CORRUPTION_FLIP_KEY:
            pass
        else:
            raise ValueError('Unknown attack: {}'.format(corruption))

        last_w = np.copy(self.model.model.optimizer.w)
        num_comm_rounds, is_updated, updates2 = self.model.update(self.updates, aggregation,
                                                    max_update_norm=MAX_UPDATE_NORM,
                                                    maxiter=maxiter,
                                                    fraction_to_discard=fraction_to_discard,
                                                    norm_bound=norm_bound,
                                                    corp_client_rate = corp_client_rate, 
                                                    mebwo_alg = mebwo_alg)
        if aggregation == AGGR_MEBWO:
            num_comm_rounds1, is_updated1 = num_comm_rounds, is_updated
            eval_loss1 = self.eval_losses_on_train_clients()
            w1, self.model.model.optimizer.w = np.copy(self.model.model.optimizer.w), last_w
            num_comm_rounds, is_updated, _ = self.model.update(self.updates, AGGR_MEBWO_OUT,
                                                        max_update_norm=MAX_UPDATE_NORM,
                                                        maxiter=maxiter,
                                                        fraction_to_discard=fraction_to_discard,
                                                        norm_bound=norm_bound,
                                                        corp_client_rate = corp_client_rate, 
                                                        weighted_updates=updates2,
                                                        mebwo_alg = mebwo_alg)
            eval_loss2 = self.eval_losses_on_train_clients()
            for i, c in enumerate(self.selected_clients):
                if c.id in corrupted_client_ids:
                    eval_loss1[i], eval_loss2[i] = eval_loss2[i], eval_loss1[i]
            if (np.asarray(eval_loss1) < np.asarray(eval_loss2)).sum() > len(eval_loss1) / 2:
                self.model.model.optimizer.reset_w(w1)
                num_comm_rounds, is_updated = num_comm_rounds1, is_updated1
            
        self.total_num_comm_rounds += num_comm_rounds
        self.updates = []
        return self.total_num_comm_rounds, is_updated

    def test_model(self, clients_to_test=None, train_and_test=True, split_by_user=False, train_users=True):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            train_and_test: If True, also measure metrics on training data
        """
        if clients_to_test is None:
            clients_to_test = self.selected_clients
            print("sever.py line201: clients_to_test = self.selected_clients")
        metrics = {}

        self.model.send_to(clients_to_test)
        for client in clients_to_test:
            c_metrics = client.test(self.model.cur_model, train_and_test, split_by_user=split_by_user, train_users=train_users)
            metrics[client.id] = c_metrics

        return metrics

    def get_clients_info(self, clients=None):
        """Returns the ids, hierarchies, num_train_samples and num_test_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients
        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_train_samples = {c.id: c.num_train_samples for c in clients}
        num_test_samples = {c.id: c.num_test_samples for c in clients}

        return ids, groups, num_train_samples, num_test_samples

    def eval_losses_on_train_clients(self, clients=None):
        # Implemented only when split_by_user is True
        losses = []

        if clients is None:
            clients = self.selected_clients

        self.model.send_to(clients)

        for c in clients:
            c_dict = c.test(self.model.cur_model, False, split_by_user=True, train_users=True)
            loss = c_dict['train_loss']
            losses.append(loss)

        return losses

    def clients_weights(self, clients=None):
        if clients is None:
            clients = self.selected_clients
        res = []
        for c in clients:
            res.append(len(c.train_data['y']))
        return res

