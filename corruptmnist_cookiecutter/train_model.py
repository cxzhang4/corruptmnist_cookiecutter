import click
import matplotlib.pyplot as plt
import torch
from models.model import SimpleCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# @click.group()
# def cli():
#     """Command line interface."""
#     pass

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=32, help="batch size to use for training")
@click.option("--epochs", default=10, help="number of epochs to train for")
def train(lr, batch_size, epochs) -> None:
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # instantiate the untrained model
    model = SimpleCNN().to(DEVICE)

    # create the dataset object from the preprocessed data
    train_images = torch.load(f"data/processed/corruptmnist_cookiecutter/train_images.pt")
    train_target = torch.load(f"data/processed/corruptmnist_cookiecutter/train_target.pt")
    train_ds = torch.utils.data.TensorDataset(train_images, train_target)
    print(train_ds.__len__())

    # create the dataloader
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

    print(train_dataloader)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}

    for epoch in range(epochs):
        # set the model to training mode (English?)
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            # x, y need to be on the same device
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    # save the model
    torch.save(model.state_dict(), "models/model.pth")

    # plot basic summary of training loss
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")

    # save the plot
    fig.savefig("reports/figures/training_statistics.png")

if __name__ == "__main__":
    train()