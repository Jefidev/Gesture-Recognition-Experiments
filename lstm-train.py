import torch
import torch.nn as nn
from utils.train_eval import train_model, eval_model
from models.VideoRNN import VideoRNN
from utils.lsfb_dataset_loader import load_lsfb_dataset
from batch_loaders.RNNBatchLoader import RNNBatchLoader
import mlflow

# Loading gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Setup ressources
params = {
    "epoch": 20,
    "batch_size": 20,
    "learning_rate": 0.001,
    "height": 144,
    "width": 180,
    "hidden_size": 2048,
    "max_frames": 14,
    "cumulation": 10,
    "lstm_layer": 2,
}


epoch = params["epoch"]
batch_size = params["batch_size"]
RUN_NAME = f"epoch:{epoch}-batch:{batch_size}"

## Loading data and setup the batch loader
data = load_lsfb_dataset(
    "/home/jeromefink/Documents/unamur/signLanguage/Data/most_frequents_25"
)
train = data[data["subset"] == "train"]
test = data[data["subset"] == "test"]

print(len(data))

loader = RNNBatchLoader(
    train, batch_size, params["max_frames"], (params["height"], params["width"])
)
val_loader = RNNBatchLoader(
    test, batch_size, params["max_frames"], (params["height"], params["width"])
)


n_class = len(loader.get_label_mapping())
net = VideoRNN(params["hidden_size"], n_class, batch_size, device, 2)
print(net)

# Chosing optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=params["learning_rate"])

# Training loop
mlflow.set_experiment("ConvNet + GRU")

with mlflow.start_run(run_name="LSFB-25 bidirectionnal LSTM"):
    mlflow.log_params(params)

    for iter in range(1, epoch + 1):
        print(f"{iter}/{epoch}\n")

        train_loss, train_acc = train_model(
            net, criterion, optimizer, loader, device, batch_size, params["cumulation"]
        )
        print(f"train_loss : {train_loss}  train_acc : {train_acc}")
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("train_acc", train_acc.item())

        eval_loss, eval_acc = eval_model(net, criterion, val_loader, device, batch_size)
        print(f"eval_loss : {eval_loss}  eval_acc : {eval_acc}")
        mlflow.log_metric("eval_loss", eval_loss)
        mlflow.log_metric("eval_acc", eval_acc.item())

