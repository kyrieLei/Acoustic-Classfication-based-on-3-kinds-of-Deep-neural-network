
import copy
from pcgrad import PCGrad
from timm.scheduler import CosineLRScheduler
from torch import nn
from torch.utils.data import DataLoader
from models_training.training import train, val
from model.model1 import CNN
from model.model2 import Cnn
from model.model3 import Model
from model.model4 import resnet
from model.model5 import Convnet

from torch.utils.data import random_split
from util.sc import *
import matplotlib.pyplot as plt
from util.func import SoundDS

if __name__=='__main__':

    conf = config()
    #model=CNN()        #最早的CNN模型
    #model=Cnn()        #改进的模型2
    #model=Model(imgSize=(64, 376), inputChannel=1, rnnHiddenSize=512, outputChannel=32, numChars=3)     #模型3
    #model=resnet()       #教师模型
    model=Convnet()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_path = "metadata/"
    data = pd.read_csv(root_path + "sound.csv")
    data['relative_path'] = '/fold' + data['fold'].astype(str) + '/' + data['file_name'].astype(str)
    data = data[['relative_path', 'classID']]

    data_path = 'metadata/audio'

    data_pet = pd.read_csv(root_path + "pet.csv")
    data_pet['relative_path'] = '/fold' + data_pet['fold'].astype(str) + '/' + data_pet['file_name'].astype(str)
    data_pet = data_pet[['relative_path', 'classID']]

    data_ring = pd.read_csv(root_path + "ring.csv")
    data_ring['relative_path'] = '/fold' + data_ring['fold'].astype(str) + '/' + data_ring['file_name'].astype(str)
    data_ring = data_ring[['relative_path', 'classID']]

    data_tap = pd.read_csv(root_path + "tap.csv")
    data_tap['relative_path'] = '/fold' + data_tap['fold'].astype(str) + '/' + data_tap['file_name'].astype(str)
    data_tap = data_tap[['relative_path', 'classID']]


    my_data = SoundDS(data, data_path)


    my_ring_data = SoundDS(data_ring, data_path)
    my_pet_data = SoundDS(data_pet, data_path)
    my_tap_data = SoundDS(data_tap, data_path)


    num_items = len(my_data)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_dataset, val_dataset = random_split(my_data, [num_train, num_val])




    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2)

    pet_dataloader=DataLoader(my_pet_data,batch_size=conf.batch_size,shuffle=False,num_workers=2)
    ring_dataloader = DataLoader(my_ring_data, batch_size=conf.batch_size, shuffle=False, num_workers=2)
    tap_dataloader = DataLoader(my_tap_data, batch_size=conf.batch_size, shuffle=False, num_workers=2)




    model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
    model.train()
    model = torch.quantization.prepare_qat(model).to(device)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)
    scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-6, warmup_t=5, warmup_lr_init=5e-5,
                                  warmup_prefix=True)
    optimizer = PCGrad(optimizer)

    max_acc = 5
    epochs = []
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []
    for t in range(conf.epochs):
        # logs = {}
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer, t, scheduler)
        val_loss, val_acc = val(val_dataloader, model, loss_fn, t)



        epochs.append(t)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        plt.subplot(1, 2, 1)
        t1, = plt.plot(epochs, train_acc_list, 'b-', label='training')
        v1, = plt.plot(epochs, val_acc_list, 'g-', label='validation')
        plt.title('accuracy')
        plt.ylabel('accuracy')
        plt.legend(handles=[t1, v1], labels=['training', 'validation'])
        plt.subplot(1, 2, 2)
        t2, = plt.plot(epochs, train_loss_list, 'b-', label='training')
        v2, = plt.plot(epochs, val_loss_list, 'g-', label='validation')
        plt.xlabel('log loss')
        plt.ylabel('loss')
        plt.legend(handles=[t2, v2], labels=['training', 'validation'])
        plt.savefig("accuracy_loss.jpg")
        print("train_acc:", train_acc)
        print("val_acc:", val_acc)
        print("train_loss:", train_loss)
        print("val_loss:", val_loss)

        if max_acc < val_acc:
            max_acc = val_acc
            torch.save(copy.deepcopy(model).state_dict(), "model/model_best.pt")

    torch.save(copy.deepcopy(model).state_dict(), "model/model_last.pt")
    print("Final Test result of pet")
    test_loss, test_acc = val(pet_dataloader, model, loss_fn, 0)
    print("test loss :  ",test_loss)
    print("test accuracy:  ",test_acc)

    print("Final Test result of tap")
    test_loss, test_acc = val(tap_dataloader, model, loss_fn, 0)
    print("test loss :  ", test_loss)
    print("test accuracy:  ", test_acc)

    print("Final Test result of ring")
    test_loss, test_acc = val(ring_dataloader, model, loss_fn, 0)
    print("test loss :  ", test_loss)
    print("test accuracy:  ", test_acc)



