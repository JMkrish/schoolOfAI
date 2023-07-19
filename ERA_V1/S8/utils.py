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


def get_test_predictions(model, test_loader, device):
    """Get test data predictions using the model

    Args:
        model (Net): Trained Model
        test_loader (Dataloader): instance of dataloader
        device (str): Which device to use cuda/cpu

    Returns:
        tuple: all predicted values and their targets
    """
    model.eval()
    test_predictions = torch.tensor([]).to(device)
    test_targets = torch.tensor([]).to(device)

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            preds = model(data).argmax(dim=1)

            test_predictions = torch.cat((test_predictions, preds), dim=0)
            test_targets = torch.cat((test_targets, targets), dim=0)

    return test_predictions, test_targets


def get_incorrrect_predictions(model, test_loader, device):
    """Get all incorrect predictions

    Args:
        model (Net): Trained model
        test_loader (DataLoader): instance of data loader
        device (str): Which device to use cuda/cpu

    Returns:
        list: list of all incorrect predictions and their corresponding details
    """
    incorrect = []

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            pred = model(data).argmax(dim=1)

            for d, t, p in zip(data, target, pred):
                if p.eq(t.view_as(p)).item() == False:
                    incorrect.append([d.cpu(), t.cpu(), p.cpu()])

    return incorrect


def prepare_confusion_matrix(predictions, targets, class_map):
    """Prepare Confusion matrix data

    Args:
        predictions (list): List of all predictions
        targets (list): List of all actule labels
        class_map (list): Class names

    Returns:
        tensor: confusion matrix for size number of classes * number of classes
    """
    stacked = torch.stack((targets, predictions), dim=1).type(torch.int64)

    classed_count = len(class_map)

    # Create temp confusion matrix
    confusion_matrix = torch.zeros(classed_count, classed_count, dtype=torch.int64)

    # Fill up confusion matrix with actual values
    for p in stacked:
        tl, pl = p.tolist()
        confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1

    return confusion_matrix
