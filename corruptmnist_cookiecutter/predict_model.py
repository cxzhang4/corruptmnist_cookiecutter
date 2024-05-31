import click
import torch
from corruptmnist_cookiecutter.models.model import SimpleCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# @click.group()
# def cli():
#     """Command line interface."""
#     pass

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

    # read in the saved checkpoint

    # load the data that you want to predict on

    # read the features

    # model.load_state_dict(torch.load("models/model.pth"))
    # model.eval()
    model.eval()
    return torch.cat([model(batch) for batch in dataloader], 0)

@click.command
@click.argument("model_path")
@click.argument("data_path")
def predict_and_save(model_path: str, data_path: str):
    # print(data_path)
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    # load data
    test_images_ds = torch.load(data_path)
    print(test_images_ds.__len__())
    test_images_ds = test_images_ds.to(DEVICE)
    test_dataloader = torch.utils.data.DataLoader(test_images_ds, batch_size=32)

    print(predict(model, test_dataloader))

if __name__ == "__main__":
    # test_img = torch.load("data/processed/corruptmnist_cookiecutter/test_images.pt")
    # dataloader = torch.utils.data.DataLoader(test_img, batch_size=32)
    # print(predict(SimpleCNN(), dataloader))
    predict_and_save()
