import click
import torch

@click.group()
def cli():
    """Command line interface."""
    pass

def standardize(images: torch.Tensor) -> torch.Tensor:
    """Standardize images."""
    return (images - images.mean()) / images.std()

@click.command()
@click.option("--raw_dir", default="data/raw", help="Path to raw data directory")
@click.option("--processed_dir", default="data/processed", help="Path to processed data directory")
def make_data(raw_dir: str, processed_dir: str):
    """Perform data preprocessing on the .pt files."""
    # TODO: figure out how to derive this from the data
    N_IMAGES = 5
    # load data
    train_images, train_target = [], []
    for i in range(N_IMAGES):
        # put the images in a list
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))

    # concatenate the list into a single tensor
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # perform data preprocessing
    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = standardize(train_images)
    test_images = standardize(test_images)

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")

@click.command()
@click.option("--raw_dir", default="data/raw", help="Path to raw data directory")
@click.option("--processed_dir", default="data/processed", help="Path to processed data directory")
def make_data_v2(raw_dir: str, processed_dir: str):
    """Perform data preprocessing on the .pt files."""
    N_IMAGES = 4
    start_idx = 6
    # load data
    train_images, train_target = [], []
    for i in range(start_idx, start_idx + N_IMAGES):
        # put the images in a list
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))

    # concatenate the list into a single tensor
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # perform data preprocessing
    train_images = train_images.unsqueeze(1).float()
    train_target = train_target.long()

    train_images = standardize(train_images)

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")

cli.add_command(make_data)
cli.add_command(make_data_v2)

if __name__ == "__main__":
    cli()