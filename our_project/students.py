import numpy
import torch

from cfg import LR, MAX_DIGITS, NUM_CHARS, OPTIM, seed_everything, SEED
from classroom import WRITER

'''Welcome to the student's site. We have an abstract class of what is a
student. He basically can "learn from task", which means training its model.
This is mainly traditional pytorch training loops. Note that data is generated
on the go by calling <task.generate_data>. The behavior of the student depends
heavily on the given task.'''

class AbstractStudent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.optimizer = None
        self.global_step = 0
        self.train_batch_step = 0
        self.train_epoch_step = 0

    def learn_from_task(self, task):
        # check that it was correctly initialized
        assert self.model is not None
        assert self.optimizer is not None

        # intialize validation variables and data
        val_data = task.generate_data(task.uniform_dist, size=task.val_size)  # originally: task.val_size)
        val_X, val_y, val_lens = val_data
        val_X = torch.from_numpy(val_X).float().to(self.device)  # in order to correctly run through the network

        # main training loop
        for n_epoch in range(task.epochs):
            # training part
            self.model.train()  # set mode 
            train_data = task.generate_data(  # generate training data
                task.train_dist, task.train_size
            )
            self._train_epoch(  # train over all the data one time
                task.loss_fn, train_data, task.batch_size
            )

            # evaluation part
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(val_X).transpose(0, 1)
                val_score = task.val_score_fn(val_pred, val_y, val_lens)

            self.global_step += 1
            WRITER.add_scalars('Student/Val_epoch_score_accperdigit', {str(i+1):ob for i, ob in enumerate(val_score)}, self.global_step)

            # # check if finished
            # if task.finished(val_score):
            #     break

        # output observation of trained model
        self.model.eval()
        observation = task.get_observation(self.model)
        return observation

    def _train_epoch(self, loss_fn, train_data, batch_size=1):
        train_X, train_y, train_lens = train_data

        # training loop
        current_ind = 0
        while current_ind < len(
            train_y
        ):  # while there is still data to be batched
            X = (
                torch.from_numpy(
                    train_X[current_ind : current_ind + batch_size]
                )
                .float()
                .to(self.device)
            )
            y = (
                torch.from_numpy(
                    train_y[current_ind : current_ind + batch_size]
                )
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

            self.train_batch_step += 1
            WRITER.add_scalar('Student/Train_batch_loss', loss.detach().item(), self.train_batch_step)
        self.train_epoch_step += 1
        WRITER.add_scalar('Student/Train_epoch_loss', loss.detach().item(), self.train_epoch_step)


'''Particular students train different models. But the problem is formulated
at task'''

class AdditionStudent(AbstractStudent):
    def __init__(self, hidden_size=128, seed=SEED):
        super().__init__()
        num_chars = 12  # We know this
        assert num_chars == NUM_CHARS
        if seed is not None:
            seed_everything(seed)
        self.model = AdditionLSTM(
            num_chars, hidden_size, MAX_DIGITS
        ).to(self.device)
        if OPTIM in ['SGD', 'sgd']:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=LR
            )  # without clipnorm
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=LR
            )  # without clipnorm



class AdditionLSTM(torch.nn.Module):
    def __init__(
        self, onehot_features=12, hidden_size=128, max_digits=15
    ):
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_digits = max_digits
        self.lstm_encode = torch.nn.LSTM(
            input_size=onehot_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.lstm_decode = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=False,
        )
        self.linear = torch.nn.Linear(
            in_features=128, out_features=onehot_features
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, encoded_input):
        '''Forward pass: pass the input through a LSTM, uses that repeated
        output as input of a second LSTM, and then for each time step applies
        a linear layer with n_chars outputs that are finally softmaxed. Output
        is probabilities because we need gradients to train!'''
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

