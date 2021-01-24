import torch
import matplotlib.pyplot as plt
import config
from LSTM import LSTM
from utils import dataloader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = config.DATA_DIR
train_X, train_Y, dev_X, dev_Y = dataloader.load_data(DATA_DIR)

batch_size = config.BATCH_SIZE
num_epochs = config.NUM_EPOCHS
initial_lr = config.LR
hidden_size = config.HIDDEN_SIZE
num_layers = config.NUM_LAYERS

model = LSTM(
    input_size=6,  # TODO : 6
    hidden_size=hidden_size,
    batch_size=batch_size,
    output_size=1,  # TODO : 1
    num_layers=num_layers
)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
criterion = torch.nn.MSELoss()

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("\n Training on GPU")
else:
    print("\n No GPU, training on CPU")

num_batches = int(train_X.shape[0] / batch_size)
num_dev_batches = int(dev_X.shape[0] / batch_size)
do_continue_train = False
val_loss_list, val_accuracy_list, epoch_list = [], [], []

print("Training ...")

for epoch in range(num_epochs):

    train_running_loss, train_acc = 0.0, 0.0
    hidden_train = None

    for i in range(num_batches):

        model.zero_grad()

        X_local_minibatch, y_local_minibatch = (
            train_X[i * batch_size: (i + 1) * batch_size, ],
            train_Y[i * batch_size: (i + 1) * batch_size, ]
        )
        X_local_minibatch = X_local_minibatch.permute(1, 0, 2)

        pred_Y, hidden_train = model(X_local_minibatch, hidden_train)
        # print(pred_Y.shape, y_local_minibatch.shape)
        if not do_continue_train:
            hidden_train = None
        else:
            h_0, c_0 = hidden_train
            h_0.detach_(), c_0.detach_()
            hidden_train = (h_0, c_0)

        loss = criterion(pred_Y, y_local_minibatch)
        loss.backward()
        optimizer.step()
        train_running_loss += loss.detach().item()

    print(
        "Epoch:  %d | MSELoss: %.4f"
        % (epoch, train_running_loss / num_batches)
    )

    print("Validation ...")
    if (epoch + 1) % 5 == 0:
        val_running_loss, val_acc = 0.0, 0.0

        # Compute validation loss, accuracy. Use torch.no_grad() & model.eval()
        with torch.no_grad():
            model.eval()
            hidden_train = None
            # model.hidden = model.init_hidden()
            for i in range(num_dev_batches):
                X_local_minibatch, y_local_minibatch = (
                    dev_X[i * batch_size: (i + 1) * batch_size, ],
                    dev_Y[i * batch_size: (i + 1) * batch_size, ],
                )

                X_local_minibatch = X_local_minibatch.permute(1, 0, 2)

                pred_Y, hidden_train = model(X_local_minibatch, hidden_train)
                val_loss = criterion(pred_Y, y_local_minibatch)

                val_running_loss += (
                    val_loss.detach().item()
                )  # unpacks the tensor into a scalar value

            model.train()  # reset to train mode after iterationg through validation data
            print(
                "Epoch:  %d | MSELoss: %.4f | Val Loss %.4f "
                % (
                    epoch,
                    train_running_loss / num_batches,
                    val_running_loss / num_dev_batches,
                )
            )
        epoch_list.append(epoch)
        val_loss_list.append(val_running_loss / num_dev_batches)

plt.plot(epoch_list, val_loss_list)
plt.xlabel("# of epochs")
plt.ylabel("MSELoss")
plt.title("LSTM: Loss vs # epochs")
plt.savefig('graph.png')
plt.show()
