import importlib
import torch
import torch.nn as nn
import random
import numpy as np
import yaml

from utils.data_utils import read_client_data
from model.config import load_model
from torch.utils.data import DataLoader

from utils.sys_utils import device_config, comm_config


class BaseClient():
    def __init__(self, id, args):
        self.id = id
        self.args = args
        #self.dataset_train = read_client_data(args.dataset, self.id, is_train=True)
        #self.dataset_test = read_client_data(args.dataset, self.id, is_train=False)
        # Datasets are loaded only when this client is used.
        self.dataset_train = None
        self.dataset_test = None
        self.loader_train = None
        self.loader_test = None

        self.train_sample_count = None
        
        self.device = args.device
        self.server = None

        self.lr = args.lr
        self.batch_size = args.bs
        self.epoch = args.epoch
        #self.model = load_model(args).to(args.device)
        self.model = load_model(args).to("cpu")
        self.loss_func = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(params=self.model.parameters(),
                                     lr=self.lr,
                                     momentum=0.9,
                                     weight_decay=1e-4)
        self.metric = {'loss': [], 'acc': []}


        #if self.dataset_train is not None:
        #    self.loader_train = DataLoader(
        #        dataset=self.dataset_train,
        #        batch_size=self.batch_size,
        #        shuffle=True,
        #        collate_fn=None,
        #        # drop_last=True
        #    )
        #if self.dataset_test is not None:
        #    self.loader_test = DataLoader(
        #        dataset=self.dataset_test,
        #        batch_size=self.batch_size,
        #        shuffle=False,
        #        collate_fn=None,
        #        # drop_last=True
        #    )

        self.training_time = None
        
    def load_data(self):
        """Load this client's datasets into RAM when needed."""

        if self.dataset_train is None:
            self.dataset_train = read_client_data(
                self.args.dataset, self.id, is_train=True
            )

            self.dataset_test = read_client_data(
                self.args.dataset, self.id, is_train=False
            )

            self.train_sample_count = len(self.dataset_train)

            self.loader_train = DataLoader(
                dataset=self.dataset_train,
                batch_size=self.batch_size,
                shuffle=True
            )

            self.loader_test = DataLoader(
                dataset=self.dataset_test,
                batch_size=self.batch_size,
                shuffle=False
            )


    def unload_data(self):
        """Release this client's datasets from RAM."""

        self.dataset_train = None
        self.dataset_test = None
        self.loader_train = None
        self.loader_test = None
        
    def move_to_device(self, device):
        """Move model and optimizer state to the requested device."""
        self.model.to(device)
        # Move optimizer state (e.g. SGD momentum buffers).
        for state in self.optim.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    
    def move_to_cpu(self):
        """Move model and optimizer state back to CPU."""
        self.move_to_device("cpu")
    
    def run(self):
        raise NotImplementedError

    def train(self):
        # === train ===
        total_loss = 0.0

        for epoch in range(self.epoch):
            for data in self.loader_train:
                X, y = self.preprocess(data)
                preds = self.model(X)
                loss = self.loss_func(preds, y)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.item()

        # === record loss ===
        self.metric['loss'] = total_loss / len(self.loader_train)

    def clone_model(self, target):
        p_tensor = target.model2tensor()
        self.tensor2model(p_tensor)

    def preprocess(self, data):
        X, y = data
        if type(X) == type([]):
            X = X[0]
        return X.to(self.device), y.to(self.device)

    def local_test(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.loader_test:
                X, y = self.preprocess(data)
                preds = self.model(X)

                _, preds_y = torch.max(preds.data, 1)
                total += y.size(0)
                correct += (preds_y == y).sum().item()
        self.metric['acc'] = 100.00 * correct / total


    def reset_optimizer(self, decay=True):
        if not decay:
            return
        for param_group in self.optim.param_groups:
            param_group['lr'] =  self.lr * (self.args.gamma ** self.server.round)

    def model2tensor(self, params=None):
        alg_module = importlib.import_module(f'alg.{self.args.alg}')
        p_keys = getattr(alg_module, 'p_keys') if hasattr(alg_module, 'p_keys') else []
        p_params = [any(key == name.split('.')[0] for key in p_keys)
                    for name, _ in self.model.named_parameters()]
        selected_params = params if params is not None else [not is_p for is_p in p_params]

        return torch.cat([param.detach().view(-1)
                          for selected, param in zip(selected_params, self.model.parameters())
                          if selected is True], dim=0)

    def tensor2model(self, tensor, params=None):
        alg_module = importlib.import_module(f'alg.{self.args.alg}')
        p_keys = getattr(alg_module, 'p_keys') if hasattr(alg_module, 'p_keys') else []
        p_params = [any(key == name.split('.')[0] for key in p_keys)
                    for name, _ in self.model.named_parameters()]
        selected_params = params if params is not None else [not is_p for is_p in p_params]

        param_index = 0
        for selected, param in zip(selected_params, self.model.parameters()):
            if selected:
                with torch.no_grad():
                    param_size = param.numel()
                    param.copy_(tensor[param_index: param_index + param_size].view(param.shape).detach())
                    param_index += param_size

    def comm_bytes(self):
        model_tensor = self.model2tensor()
        return model_tensor.numel() * model_tensor.element_size()

class BaseServer(BaseClient):
    def __init__(self, id, args, clients):
        super().__init__(id, args)
        self.client_num = args.total_num
        self.sample_rate = args.sr
        self.total_round = args.rnd

        self.clients = clients
        self.sampled_clients = []
        self.round = 0
        self.wall_clock_time = 0
        self.received_params = []

        delays = device_config(self.client_num)
        bandwidths = comm_config(self.client_num)

        for client in self.clients:
            client.server = self
            client.delay = delays[client.id]
            client.bandwidth = bandwidths[client.id]

    def run(self):
        raise NotImplementedError

    def sample(self):
        sample_num = int(self.sample_rate * self.client_num)
        self.sampled_clients = sorted(random.sample(self.clients, sample_num), key=lambda x: x.id)

    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        for client in self.sampled_clients:
            client.clone_model(self)

    def client_update(self):
        for client in self.sampled_clients:
            try:
                # Load only the active client's data.
                client.load_data()
                
                #Move only the active client's model to GPU
                client.move_to_device(client.device)
                client.model.train()
                client.reset_optimizer()
                client.run()
                
            finally:
                #Keep the model nd optimizer state on CPU between clients.
                client.move_to_cpu()
                client.unload_data()
            	
        self.wall_clock_time += max([client.training_time for client in self.sampled_clients])

    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        def nan_to_zero(tensor):
            return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        self.received_params = [nan_to_zero(client.model2tensor()) for client in self.sampled_clients]

    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        total_samples = sum(client.train_sample_count for client in self.sampled_clients)
        weights = [client.train_sample_count / total_samples for client in self.sampled_clients]

        self.received_params = [params * weight for weight, params in zip(weights, self.received_params)]
        avg_tensor = sum(self.received_params)
        self.tensor2model(avg_tensor)

    def test_all(self):
        self.metric['acc'] = []
        for client in self.clients:
            try:
                client.load_data()
                client.clone_model(self)
                client.move_to_device(client.device)
                client.local_test()
                self.metric['acc'].append(client.metric['acc'])
            finally:
                client.move_to_cpu()
                client.unload_data()

        return {
            'acc': np.mean(self.metric['acc']),
            'acc_std': np.std(self.metric['acc']),
        }
