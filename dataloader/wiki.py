import os
import re
import json

from torch.utils.data import Dataset, DataLoader, random_split

class WikiDataset(Dataset):

  def __init__(self, path="", prefix="train"):

    assert os.path.isdir(path)

    self.documents = []
    filename_list = os.listdir(path)
    for file in filename_list:
      path_to_file = os.path.join(path, file)
      if not os.path.isfile(path_to_file):
        continue
      self.documents.append(path_to_file)

  def __len__(self):
    """ Returns the number of documents. """
    return len(self.documents)

  def __getitem__(self, idx):
    document_path = self.documents[idx]
    document_name = document_path.split("/")[-1]

    items = []

    with open(document_path, encoding="utf-8") as source:
      raw_text = source.readlines()
      for obj in raw_text:
        text = json.loads(obj)['text']
        text = re.sub('\\n', ' ', text)
        text = re.sub('\\s+', ' ', text)
        items.append(text)

    return items