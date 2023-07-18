import torch
import torch.optim as optim


def get_device() -> tuple:
    """Get Device type

    Returns:
        tuple: Device type
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return (use_cuda, device)


def get_data_loader(ds, kwargs):
    return torch.utils.data.DataLoader(ds, **kwargs)


def get_lr_scheduler(optimizer, step_size=15, gamma=0.1, verbose=True):
    return optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma, verbose=verbose
    )
