import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import optim as optim
from torch.utils.data import DataLoader

from model import Net
from utils import get_lr_scheduler
from utils import get_test_predictions
from utils import get_incorrrect_predictions
from visualize import plot_incorrect_predictions
from utils import prepare_confusion_matrix
from visualize import plot_confusion_matrix
from visualize import plot_network_performance


def get_sgd_optimizer(model, lr, momentum=0, weight_decay=0):
    return optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )


class ModelTrainPipeline:
    def __init__(
        self,
        criteria,
        train_loader,
        test_loader,
        epochs,
        device,
        norm="none",
        dropout=0.0,
        lr=0.01,
        momentum=0.0,
        step_size=15,
        gamma=0.1,
        skip_connets=False,
        model_summary=False,
    ):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

        self.device = device
        self.criteria = criteria
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.norm = norm
        self.dropout = dropout
        self.lr = lr
        self.momentum = momentum
        self.step_size = step_size
        self.gamma = gamma
        self.skip_connets = skip_connets

        self.model = Net(norm, dropout, skip_connets).to(device)

        if model_summary:
            image, _ = train_loader.dataset[0]
            self.model.model_summary(tuple(image.shape))

        self.optimizer = get_sgd_optimizer(
            self.model, lr=self.lr, momentum=self.momentum
        )
        self.scheduler = get_lr_scheduler(
            self.optimizer, step_size=self.step_size, gamma=self.gamma
        )

    def train(self):
        self.model.train()
        pbar = tqdm(self.train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Predict
            pred = self.model(data)

            # Calculate loss
            loss = self.criteria(pred, target)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            correct += self.GetCorrectPredCount(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f"Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
            )

        self.train_acc.append(100 * correct / processed)
        self.train_losses.append(train_loss / len(self.train_loader))

    def test(self):
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += self.criteria(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss

                correct += self.GetCorrectPredCount(output, target)

        test_loss /= len(self.test_loader.dataset)
        self.test_acc.append(100.0 * correct / len(self.test_loader.dataset))
        self.test_losses.append(test_loss)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(self.test_loader.dataset),
                100.0 * correct / len(self.test_loader.dataset),
            )
        )

    def run(self):
        print(
            f"Device: {self.device}\nEpochs: {self.epochs}\n"
            f"Lr: {self.lr}\nBatch Size: {self.train_loader.batch_size}\n"
            f"Dropout: {self.dropout}\nMomentum: {self.momentum}"
        )
        print("----------------------------------------------------------------")
        print(f"Normalization: {self.norm}\n" f"Skip Connection: {self.skip_connets}")
        print("----------------------------------------------------------------")
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}")
            self.train()
            self.test()
            self.scheduler.step()
        print("****************************************************************")

    # Open Neural Network Exchange(ONNX)
    def save_model_as_onnx(self):
        image, _ = self.train_loader.dataset[0]

        # Add a batch dimension to the input image
        image = torch.unsqueeze(image, 0)

        torch.onnx.export(
            self.model,
            image.to(self.device),
            f"norm_{self.norm}-lr_{self.lr}-model.onnx",
            input_names=["image"],
            output_names=["classes"],
        )

    def print_performance(self, class_map):
        plot_network_performance(
            self.epochs,
            self.train_losses,
            self.test_losses,
            self.train_acc,
            self.test_acc,
        )

        # Confusion Matrix
        test_preds, test_targets = get_test_predictions(
            self.model, self.test_loader, self.device
        )
        confusion_matrix = prepare_confusion_matrix(test_preds, test_targets, class_map)
        plot_confusion_matrix(confusion_matrix, class_map)

        # Misclassified Images
        wrong_predicts = get_incorrrect_predictions(
            self.model, self.test_loader, self.device
        )
        plot_incorrect_predictions(wrong_predicts, class_map, count=10)

    def GetCorrectPredCount(self, pPrediction, pLabels):
        return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
