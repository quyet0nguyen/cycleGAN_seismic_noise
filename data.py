import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch 

def load_dataset(link):
    data_train = np.load(link)
    
    #== rerange of the input to 0, 1
    max_val = np.amax(data_train)
    min_val = np.amin(data_train)
    data_train = (data_train - min_val)/ ( max_val - min_val)

    #== normalize data
    # mean  =  np.mean(data_train)
    # std = np.std(data_train)
    # data_train = (data_train - mean) / std

    return data_train

def generate_image_seismic(root_img, image_size, batch_size, num_iter):
    X_train = np.empty((batch_size*num_iter,image_size,image_size))
    #fill minimum number in dataset
    #X_train.fill(0)
    
    for _ in range(batch_size*num_iter):
        x = random.randrange(root_img.shape[0]-image_size)
        y = random.randrange(root_img.shape[1]-image_size)
        z = random.randrange(root_img.shape[2])

        X_train[_,:,:] = root_img[x:x+image_size, y:y+image_size,z]
    
    return X_train

def generate_image_noise(seismic_img, image_size, batch_size, num_iter):
    noised_train_data = np.empty((batch_size*num_iter,image_size,image_size))
    noised_train_data[:,:,:] = seismic_img[:,:,:]
    # max_val = np.amax(noised_train_data)
    # min_val = np.amin(noised_train_data)
    #fill minimum number in dataset
    for _ in range(noised_train_data.shape[0]):
      size_noise = random.randrange(2,10)
      ran_noise = random.randrange(noised_train_data.shape[1]-size_noise)
      #noised_train_data[_,:,ran_noise:ran_noise+size_noise] = np.random.rand(noised_train_data.shape[1], size_noise)
      noised_train_data[_,:,ran_noise:ran_noise+size_noise] = np.zeros((noised_train_data.shape[1], size_noise))

    return noised_train_data

def reshape_data(dataset):
    dataset = dataset.reshape(dataset.shape[0],
                            1,
                            dataset.shape[1],
                            dataset.shape[2])

    return dataset 

def data_loader(dataset, batch_size):
    dataset = torch.tensor(dataset)
    dataset = TensorDataset(dataset)
    dataset = DataLoader(dataset, batch_size = batch_size)

    return dataset

def train_dataset(dir, batch_size, image_size, num_iter):
    dataset = load_dataset(dir)

    seismic_img = generate_image_seismic(dataset, image_size, batch_size, num_iter)
    noise_img = generate_image_noise(seismic_img, image_size, batch_size, num_iter)

    seismic_img = reshape_data(seismic_img)
    seismic_img = data_loader(seismic_img, batch_size)

    noise_img = reshape_data(noise_img)
    noise_img = data_loader(noise_img, batch_size)

    return seismic_img, noise_img
    
if __name__ == "__main__":
    dir = '/content/AiCrowdData/data_train/data.npy'
    batch_size = 64
    image_size = 32
    num_iter = 900
