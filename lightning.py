import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

class LitMNIST(pl.LightningModule):
  def __init__(self, hidden_size=64, learning_rate=2e-4):
    super().__init__()

    # mnist images are (1, 28, 28) (channels, width, height)
    self.layer_1 = nn.Linear(28 * 28, 128)
    self.layer_2 = nn.Linear(128, 256)
    self.layer_3 = nn.Linear(256, 10)

  def forward(self, x):
    batch_size, channels, width, height = x.size()

    # (b, 1, 28, 28) -> (b, 1*28*28)
    x = x.view(batch_size, -1)
    x = self.layer_1(x)
    x = F.relu(x)
    x = self.layer_2(x)
    x = F.relu(x)
    x = self.layer_3(x)

    x = F.log_softmax(x, dim=1)
    return x
  
  def training_step(self, batch, batch_idx):
      x, y = batch
      logits = self(x)
      loss = F.nll_loss(logits, y)
      return loss
    
  def validation_step(self, batch, batch_idx):
      x, y = batch
      logits = self(x)
      loss = F.nll_loss(logits, y)
      preds = torch.argmax(logits, dim=1)
      acc = accuracy(preds, y)
      self.log('val_loss', loss, prog_bar=True)
      self.log('val_acc', acc, prog_bar=True)
      return loss
      
  def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
      return optimizer
  
    
class MNISTDataModule(pl.LightningDataModule):
  def __init__(self, batch_size=64):
      super().__init__()
      self.batch_size = batch_size

  def prepare_data(self):
      MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
      MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

  def setup(self, stage=None):
      # transform
      transform=transforms.Compose([transforms.ToTensor()])
      mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transform)
      mnist_test = MNIST(os.getcwd(), train=False, download=False, transform=transform)

      # train/val split
      mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

      # assign to use in dataloaders
      self.train_dataset = mnist_train
      self.val_dataset = mnist_val
      self.test_dataset = mnist_test

  def train_dataloader(self):
      return DataLoader(self.train_dataset, batch_size=self.batch_size)

  def val_dataloader(self):
      return DataLoader(self.val_dataset, batch_size=self.batch_size)

  def test_dataloader(self):
      return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
dm = MNISTDataModule()
model = LitMNIST()
trainer = pl.Trainer(
    max_epochs=2,
    gpus=AVAIL_GPUS,
    progress_bar_refresh_rate=20,
)
trainer.fit(model, dm)