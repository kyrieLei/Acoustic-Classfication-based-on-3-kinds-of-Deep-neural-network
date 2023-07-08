from torch.utils.data import DataLoader

from torch.utils.data import random_split
from util.sc import *

from util.func import SoundDS




device = "cuda" if torch.cuda.is_available() else "cpu"
root_path = "metadata/"
data = pd.read_csv(root_path + "sound.csv")
data['relative_path'] = '/fold' + data['fold'].astype(str) + '/' + data['file_name'].astype(str)
data = data[['relative_path', 'classID']]

data_path = 'metadata/audio'




myds = SoundDS(data, data_path)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

for i,data in enumerate(train_dl):
  print(data[0].shape)

  #print(data[1])