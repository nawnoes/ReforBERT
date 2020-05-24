from torch.utils.data import Dataset, DataLoader, random_split
import logging


def build_dataloaders(dataset, batch_size, train_test_split=0.1, train_shuffle=True, eval_shuffle=True):
  """
  Builds the Training and Eval DataLoaders

  :param train_test_split: The ratio split of test to train data.
  :param train_shuffle: (bool) True if you wish to shuffle the train_dataset.
  :param eval_shuffle: (bool) True if you wish to shuffle the eval_dataset.
  :return: train dataloader and evaluation dataloader.
  """
  # 데이터셋 길이
  dataset_len = len(dataset)

  # 학습, 평가 데이터 나누기
  eval_len = int(dataset_len * train_test_split)
  train_len = dataset_len - eval_len

  train_dataset, eval_dataset = random_split(dataset, (train_len, eval_len))

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
  eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=eval_shuffle)


  logging.info(f'''train_dataloader size: {len(train_loader.dataset)} | shuffle: {train_shuffle}
                     eval_dataloader size: {len(eval_loader.dataset)} | shuffle: {eval_shuffle}''')

  return train_loader, eval_loader