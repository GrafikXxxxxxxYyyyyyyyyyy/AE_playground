import time
import torch
import wandb
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from torchvision.utils import make_grid



class Logger ():
    def __init__ (self, 
                  test_dataset, 
                  loss_function,
                  metric=None,
                  delimeter=100):
        
        self.step = 0
        self.delim = delimeter
        
        self.dataset = test_dataset
        self.loss_function = loss_function
        self.metric = metric
        
        # Задаём таблицы, которые хотим логгировать
        self.intermediate_results = wandb.Table(["Step", "Real images", "Reconstructed images", "MSE loss"])
        
        return
    
    
    
    def __call__ (self, model, train_loss):
        self.step += 1
        
        # Логгируем ошибку на батче тренировочной выборки
        wandb.log({"Loss on TRAIN": train_loss})
        
        # Каждые delim шагов логгируем результаты, полученные на тестовой выборке 
        if self.step % self.delim == 0:
            test_loss = 0
            total_len = 0
            metric_loss = 0
            for x_batch_test, _ in self.dataset:
                output = model(x_batch_test)
                
                if self.metric is not None:
                    metric_loss += self.metric(output, x_batch_test).cpu().item()*len(x_batch_test)
                
                test_loss += self.loss_function(output, x_batch_test).cpu().item()*len(x_batch_test)
                total_len += len(x_batch_test)
                
            
            # Если задана метрика, то логгируем
            if self.metric is not None:
                wandb.log({"Metric": metric_loss / total_len})
            
            # Логгируем ошибку на тестовой выборке
            wandb.log({"Loss on TEST": test_loss / total_len})

            x_batch_test, _ = next(iter(self.dataset))
            real = make_grid(x_batch_test[ :min(len(x_batch_test), int(4**2))], 4).permute(1,2,0)
            reconstructed = model(x_batch_test[ :min(len(x_batch_test), int(4**2))]).detach()
            reconstructed = make_grid(reconstructed, 4).permute(1,2,0)
            
            # Добавляем запись в таблицу
            self.intermediate_results.add_data (self.step, 
                                                wandb.Image(real.numpy()), 
                                                wandb.Image(reconstructed.numpy()), 
                                                test_loss / total_len)
            
            # Сохраняем структуру самой модели
            torch.onnx.export(model, x_batch_test, "model.onnx")
            wandb.save("model.onnx")
        
        return
    

    def stop_logging (self):
        """
        Данная функция просто завершает процесс логирования и выводит всю информацию,
        которая копилась в процессе (таблицы, onnx и пр.)
        """
        # В нашем случае просто логгируем табличку с промежуточными картинками
        wandb.log({"Intermediate results": self.intermediate_results})

        return 



# Функция обучения на отдельном батче
def train_batch (x_batch, model, optimizer, loss_function):
    # Переводим модель в режим обучения
    model.train()
    
    # Обнуляем остатки градиентов с прошлых шагов
    optimizer.zero_grad()
    
    # Строим прогноз по батчу
    output = model(x_batch)
    
    # Ошибка между output и x_batch, т.к AE
    loss = loss_function(output, x_batch)
    loss.backward()
    
    # Делаем шаг оптимизатора
    optimizer.step()
    
    return loss.cpu().item()



# Функция обучения на одной эпохе
def train_epoch (train, model, optimizer, loss_function, logger=None):
    epoch_loss = 0
    total_len = 0
    
    # цикл по генератору train (по батчам)
    for it, (x_batch, _) in enumerate(train):
        # Вызов train_batch
        batch_loss = train_batch(x_batch, model, optimizer, loss_function)
    
        # Логгируем модель если указан логгер
        if logger is not None:
            with torch.no_grad():
                logger(model, batch_loss)
        
        epoch_loss += batch_loss*len(x_batch)
        total_len += len(x_batch)

    return epoch_loss / total_len



# Функция для обучения модели
def trainer (train, model, optimizer, loss_function, epochs=5, lr=1e-3, logger=None):
    # Загрузим параметры модели в оптимизатор
    optim = optimizer(model.parameters(), lr)
    
    tqdm_iterations = tqdm(range(epochs), desc='Epoch')
    tqdm_iterations.set_postfix({'Current train loss': np.nan})

    for epoch in tqdm_iterations:
        current_loss = train_epoch(train, model, optim, loss_function, logger)
        
        tqdm_iterations.set_postfix({'Current train loss': current_loss})

    # Выводим всякую промежуточную информацию из логов
    if logger is not None:
        logger.stop_logging()        

    return