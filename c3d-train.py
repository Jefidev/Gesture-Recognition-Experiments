# https://towardsdatascience.com/pytorch-step-by-step-implementation-3d-convolution-neural-network-8bf38c70e8b3

from batch_loaders.C3DLoader import C3DLoader
from utils.lsfb_dataset_loader import load_lsfb_dataset
from models.C3D import C3D
import mlflow
import torch.nn as nn
import torch
from utils.train_eval import train_model, eval_model

# Loading gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

path = "/home/jeromefink/Documents/unamur/signLanguage/Data/most_frequents_25"
# path = "./mock-data"
params = {
    "batch_size": 13,
    "height": 144,
    "width": 180,
    "max_frames": 20,
    "epochs": 20,
    "lr": 0.001,
    "dataset": path.split("/")[-1],
    "cumulation": 10,
}


## Loading data and setup the batch loader
#
data = load_lsfb_dataset(path)

train = data[data["subset"] == "train"]
test = data[data["subset"] == "test"]

loader = C3DLoader(
    train,
    params["batch_size"],
    params["max_frames"],
    (params["height"], params["width"]),
)

val_loader = C3DLoader(
    test,
    params["batch_size"],
    params["max_frames"],
    (params["height"], params["width"]),
)


params["n_class"] = len(loader.get_label_mapping())
net = C3D(params["n_class"])
print(net)

# Chosing optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(), lr=params["lr"], momentum=0.9, weight_decay=0.0000001
)
lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


# Training loop
mlflow.set_experiment("C3D")

with mlflow.start_run(run_name=params["dataset"]):
    mlflow.log_params(params)

    for iter in range(1, params["epochs"] + 1):
        epochs = params["epochs"]
        print(f"{iter}/{epochs}\n")

        train_loss, train_acc = train_model(
            net,
            criterion,
            optimizer,
            lr_sched,
            loader,
            device,
            params["batch_size"],
            params["cumulation"],
        )
        print(f"train_loss : {train_loss}  train_acc : {train_acc}")
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("train_acc", train_acc.item())

        eval_loss, eval_acc = eval_model(
            net, criterion, val_loader, device, params["batch_size"]
        )
        print(f"eval_loss : {eval_loss}  eval_acc : {eval_acc}")
        mlflow.log_metric("eval_loss", eval_loss)
        mlflow.log_metric("eval_acc", eval_acc.item())

