import numpy
from numpy.lib.arraysetops import isin
import torch

from cfg import LR, MAX_DIGITS, NUM_CHARS, OPTIM, seed_everything, SEED, DEVICE, LOG_DICT
from classroom import WRITER

"""Welcome to the student's site. We have an abstract class of what is a
student. He basically can "learn from task", which means training its model.
This is mainly traditional pytorch training loops. Note that data is generated
on the go by calling <task.generate_data>. The behavior of the student depends
heavily on the given task."""


class Logger:
    def __init__(self, log_dict):
        self.active = log_dict["activate"]
        self.size = log_dict["size"]
        self.freq = log_dict["freq"]
        self.dist = log_dict.get("dist", [1 / MAX_DIGITS] * MAX_DIGITS)
        self.random_data = log_dict["regenerate_data"]
        self.device = DEVICE

    def get_data(self, task):
        if self.random_data:
            return self._create_data(task)
        else:
            if hasattr(self, "data"):
                return self.data
            else:
                self.data = self._create_data(task)
                return self.data

    def _create_data(self, task):
        log_data = task.generate_data(
            self.dist, size=self.size
        )  # originally: task.val_size)
        log_X, log_y, log_lens = log_data
        log_X = (
            torch.from_numpy(log_X).float().to(self.device)
        )  # in order to correctly run through the network
        return log_X, log_y, log_lens

    def log(self, loss, count, model, task):
        if self.active and count % self.freq == 0:
            log_X, log_y, log_lens = self.get_data(task)
            model.eval()
            with torch.no_grad():
                log_pred = model(log_X).transpose(0, 1)
                log_scores = self.logging_fn(log_pred, log_y, log_lens, task)

            for name, score in log_scores.items():
                try:
                    if len(score) > 1:
                        WRITER.add_scalars(
                            f"Student/logs_{name}",
                            {str(i + 1): ob for i, ob in enumerate(score)},
                            count,
                        )
                except TypeError:
                    WRITER.add_scalar(
                        f"Student/logs_{name}",
                        score,
                        count,
                    )

    def logging_fn(self, pred, y, lens, task):
        scores = {}
        y = y * 1
        pred = pred.numpy()
        with torch.no_grad():
            scores["acc_per_len"] = task.accuracy_per_length(pred, y, lens)
            scores["full_n_acc"] = task.full_number_accuracy(torch.tensor(pred), torch.tensor(y))
            scores["loss_fn"] = task.loss_fn(torch.tensor(pred), torch.tensor(y), lens)
            scores["per_digit_loss"] = task.per_digit_loss(torch.tensor(pred), y, lens)
        return scores


class AbstractStudent:
    def __init__(self):
        self.device = DEVICE
        self.model = None
        self.optimizer = None
        self.update_count = 0
        self.logger = Logger(LOG_DICT)

    def learn_from_task(self, task):
        # check that it was correctly initialized
        assert self.model is not None
        assert self.optimizer is not None

        # intialize validation variables and data
        if self.logger.active:
            log_X, log_y, log_lens = self.logger.get_data(task)

        # main training loop
        for n_epoch in range(task.epochs):
            # training part
            self.model.train()  # set mode
            train_data = task.generate_data(  # generate training data
                task.train_dist, task.train_size
            )
            task.last_loss = self._train_epoch(  # train over all the data one time
                task.loss_fn, train_data, task.batch_size, task
            )

        # output observation of trained model
        self.model.eval()
        observation = task.get_observation(self.model)
        return observation

    def _train_epoch(self, loss_fn, train_data, batch_size=1, task=None):
        train_X, train_y, train_lens = train_data

        # training loop
        current_ind = 0
        while current_ind < len(train_y):  # while there is still data to be batched
            X = (
                torch.from_numpy(train_X[current_ind : current_ind + batch_size])
                .float()
                .to(self.device)
            )
            y = (
                torch.from_numpy(train_y[current_ind : current_ind + batch_size])
                .float()
                .to(self.device)
            )
            lens = train_lens[current_ind : current_ind + batch_size]
            current_ind += batch_size

            # update weights
            self.optimizer.zero_grad()
            pred = self.model(X)
            if len(pred.shape) == 2:
                pred = pred.unsqueeze(1)
            pred = pred.transpose(0, 1)
            loss = loss_fn(pred, y, lens)
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

            self.update_count += 1
            self.logger.log(loss.detach().item(), self.update_count, self.model, task)


        return loss.detach()


"""Particular students train different models. But the problem is formulated
at task"""


class AdditionStudent(AbstractStudent):
    def __init__(self, hidden_size=128, seed=SEED):
        super().__init__()
        num_chars = 12  # We know this
        assert num_chars == NUM_CHARS
        if seed is not None:
            seed_everything(seed)
        self.model = AdditionLSTM(num_chars, hidden_size, MAX_DIGITS).to(self.device)
        if OPTIM in ["SGD", "sgd"]:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=LR
            )  # without clipnorm
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=LR
            )  # without clipnorm


class AdditionBanditStudent(AdditionStudent):
    def learn_from_task(self, task):
        # check that it was correctly initialized
        assert self.model is not None
        assert self.optimizer is not None
        assert task.reward_name == "PG"  # only to be used with prediction gain
        assert (
            task.epochs == task.train_size == task.batch_size == 1
        )  # epochs and batch size is redundant

        # training part
        self.model.train()  # set mode
        train_data = task.generate_data(  # generate training data
            task.train_dist, task.train_size
        )

        train_X, train_y, train_lens = train_data
        X = torch.from_numpy(train_X).float().to(self.device)
        y = torch.from_numpy(train_y).float().to(self.device)
        lens = train_lens

        # update weights
        self.optimizer.zero_grad()
        pred = self.model(X)
        if len(pred.shape) == 2:
            pred = pred.unsqueeze(1)
        pred = pred.transpose(0, 1)
        loss = task.loss_fn(pred, y, lens)
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        self.optimizer.step()

        self.update_count += 1
        self.logger.log(loss.detach().item(), self.update_count, self.model)

        # output observation of trained model
        observation = task.get_observation(self.model, loss.detach(), X, y, lens)
        return observation


class AdditionLSTM(torch.nn.Module):
    def __init__(self, onehot_features=12, hidden_size=128, max_digits=15):
        """
        onehot_features is 12 because len('0123456789+ ') = 12

        Creates encoding layer. It will take as input (batch, seq, feature) and
        will output (batch, hidden_dim) values (we will use h_n)

        Creates decoding layer. It will take as input (seq=max_digits+1, batch, hidden_dim).
        The output is h_ts of shape (max_digits+1, 1, hidden_dim).

        Finally a linear layer applied in the time dimension: it takes 
        (max_digits+1, hidden_dim) and outputs (max_digits+1, onehot_features).
        + Softmax
        """
        super().__init__()
        self.device = DEVICE
        self.max_digits = max_digits
        self.lstm_encode = torch.nn.LSTM(
            input_size=onehot_features, hidden_size=hidden_size, batch_first=True,
        )
        self.lstm_decode = torch.nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=False,
        )
        self.linear = torch.nn.Linear(in_features=128, out_features=onehot_features)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, encoded_input):
        """Forward pass: pass the input through a LSTM, uses that repeated
        output as input of a second LSTM, and then for each time step applies
        a linear layer with n_chars outputs that are finally softmaxed. Output
        is probabilities because we need gradients to train!"""
        h_ts, (h_n, c_n) = self.lstm_encode(encoded_input)
        expanded_h_n = h_n.expand(
            (self.max_digits + 1, h_n.shape[1], h_n.shape[2])
        ).contiguous()  # creates a brand new tensor
        h_ts, (h_n, c_n) = self.lstm_decode(
            expanded_h_n
        )  # we change batch order: (seq, batch, feature)
        x = self.linear(
            torch.squeeze(h_ts)
        )  # squeeze in order to compute softmax in the correct axis
        x = self.softmax(x)
        return x

