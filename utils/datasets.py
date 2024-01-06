import os
import cv2
import torch
import shutil
import subprocess
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import transforms

from torch.utils.data import random_split, DataLoader
from torchvision.datasets import DatasetFolder, ImageFolder



# Создаёт директорию с данными (если её нет)
def create_dataset_folder(attrs_name="lfw_attributes.txt",
                          images_name="lfw-deepfunneled",
                          output_dir="data/fake_class"):
    
    if not os.path.exists(output_dir):
        #download if not exists
        if not os.path.exists(f"data/{images_name}"):
            print("images not found, donwloading...")
            # Качает архив с фотками
            os.system("curl -o data/tmp.tgz http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz")
            print("extracting...")
            os.system("tar xvzf data/tmp.tgz -C data/ && rm data/tmp.tgz")
            print("done")
            assert os.path.exists(f"data/{images_name}")

        if not os.path.exists(f"data/{attrs_name}"):
            print("attributes not found, downloading...")
            # Получаем атрибуты
            os.system(f"curl -o data/{attrs_name} http://www.cs.columbia.edu/CAVE/databases/pubfig/download/{attrs_name}")
            print("done")
        
        # read attrs
        df_attrs = pd.read_csv(f"data/{attrs_name}",sep='\t',skiprows=1,)
        df_attrs = pd.DataFrame(df_attrs.iloc[:,:-1].values, columns = df_attrs.columns[1:])

        #read photos
        photo_ids = []
        for dirpath, dirnames, filenames in os.walk(f"data/{images_name}"):
            for fname in filenames:
                if fname.endswith(".jpg"):
                    fpath = os.path.join(dirpath,fname)
                    # Получает список токенов
                    photo_id = fname[:-4].replace('_',' ').split()
                    person_id = ' '.join(photo_id[:-1])
                    # получает номер фотографии человека как последий элемент из списка токенов
                    photo_number = int(photo_id[-1])
                    photo_ids.append({'person':person_id,'imagenum':photo_number,'photo_path':fpath})
        
        photo_ids = pd.DataFrame(photo_ids)

        #mass-merge
        df = pd.merge(df_attrs,photo_ids,on=('person','imagenum'))

        assert len(df)==len(df_attrs),"lost some data when merging dataframes"

        # create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # save photos to the output directory
        for index, row in df.iterrows():
            # Читаем изображение по его расположению
            img = cv2.imread(row['photo_path'])
            output_path = os.path.join(output_dir, f"{index}.jpg")
            cv2.imwrite(output_path, img)

        print(f"Images saved to {output_dir}")
        
        # remove the original images directory
        os.system(f"rm -rf data/{images_name}")
        print(f"Original images directory {images_name} removed.")

    return



def make_loaders (path="data/", img_size=128, test_size=0.2, batch_size=16):
    # Задаём преобразование для изображений
    tranformation = transforms.Compose([
        lambda x: np.array(x) / 255.,
        lambda x: torch.tensor(x, dtype=torch.float32).permute((2, 0, 1)),
        transforms.CenterCrop((img_size, img_size))
    ])

    # Создаём датасет
    dataset = DatasetFolder("data/", loader=torchvision.datasets.folder.default_loader, extensions='.jpg', transform=tranformation)

    # Делим на трейн/тест
    train, test = random_split(dataset, [1-test_size, test_size])

    # Создаём генераторы батчей
    train_generator = DataLoader(train, batch_size, shuffle=True)
    test_generator = DataLoader(test, batch_size, shuffle=True)

    return train_generator, test_generator