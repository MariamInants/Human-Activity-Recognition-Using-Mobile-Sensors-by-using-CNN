import tensorflow as tf
# from data_loader import *
from abc import abstractmethod
import math
import numpy as np
import random
from numpy import array
import os 
import pandas as pd

from scipy import stats
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split

import pickle

class BaseNN:
    def __init__(self, train_images_dir, val_images_dir, test_images_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name):

       
        self.train_batch_size=train_batch_size
        # self. val_batch_size= val_batch_size
        # self. test_batch_size= test_batch_size

        self.learning_rate=learning_rate
        self.width_of_image=width_of_image
        self.height_of_image=height_of_image
        self.num_classes=num_classes
        self.max_to_keep=max_to_keep
        self.model_name=model_name
        self.base_dir=base_dir
        self.num_epochs=num_epochs
        self.num_channels=num_channels


# This is for parsing the X data, you can ignore it if you do not need preprocessing
    def format_data_x(self,datafile):
        x_data = None
        for item in datafile:
            item_data = np.loadtxt(item, dtype=np.float)
            if x_data is None:
                x_data = np.zeros((len(item_data), 1))
            x_data = np.hstack((x_data, item_data))
        x_data = x_data[:, 1:]
        print(x_data.shape)
        X = None
        for i in range(len(x_data)):
            row = np.asarray(x_data[i, :])
            row = row.reshape(9, 128).T
            if X is None:
                X = np.zeros((len(x_data), 128, 9))
            X[i] = row
        print(X.shape)
        return X

# This is for parsing the Y data, you can ignore it if you do not need preprocessing
    def format_data_y(self,datafile):
        data = np.loadtxt(datafile, dtype=np.int) - 1
        YY = np.eye(6)[data]
        return YY


# Load data function, if there exists parsed data file, then use it
# If not, parse the original dataset from scratch
    def load_data(self):
        if os.path.isfile('data/data_har.npz') == True:
            data = np.load('data/data_har.npz')
            X_train = data['X_train']
            Y_train = data['Y_train']
            X_test = data['X_test']
            Y_test = data['Y_test']
        else:
        # This for processing the dataset from scratch
        # After downloading the dataset, put it to somewhere that str_folder can find
        # str_folder = 'Your root folder' + 'UCI HAR Dataset/'
            str_folder ='/Users/mariam/Desktop/ASDS/Deep_Learning/project/CNN1/UCI HAR Dataset/'
            INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]
            str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
                           INPUT_SIGNAL_TYPES]
            str_test_files = [str_folder + 'test/' + 'Inertial Signals/' + item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
            str_train_y = str_folder + 'train/y_train.txt'
            str_test_y = str_folder + 'test/y_test.txt'
            RANDOM_SEED = 10

            X_train = self.format_data_x(str_train_files)
            X_test = self.format_data_x(str_test_files)
            Y_train = self.format_data_y(str_train_y)
            Y_test = self.format_data_y(str_test_y)
            X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.3, random_state=RANDOM_SEED)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test
        





    def create_network(self):
        # tf.reset_default_graph()
     
        self.x = tf.placeholder(tf.float32, [None, self.height_of_image, self.width_of_image,9], name="X")
        self.y = tf.placeholder(tf.float32, [None, self.num_classes], name="y")

    
        self.prediction=self.network(self.x)

        self.loss = self.metrics(self.y,self.prediction)[0]
        self.accuracy=self.metrics(self.y,self.prediction)[1]
        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
        
    def load(self):
        print(" [*] Reading checkpoint...")
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        checkpoint_dir = os.path.join(os.getcwd(), self.base_dir, self.model_name, 'chekpoints')
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(ckpt_name)
            return True
        else:
            return False

    def initialize_network(self):
        self.sess= tf.InteractiveSession()
        if os.path.exists(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'))== False:
            self.sess.run(tf.global_variables_initializer())
           
        else:
            self.load()
           
        

    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = self.load_data()
        
        

        train_count = len(self.X_train)
        total_batch = len(self.X_train) // self.train_batch_size

        for epoch in range(1, self.num_epochs + 1):
            for j in range(total_batch):
                x_train_batch, y_train_batch = self.X_train[j * self.train_batch_size: self.train_batch_size * (j + 1)], \
                                               self.Y_train[j * self.train_batch_size: self.train_batch_size * (j + 1)]
                x_train_batch = np.reshape(x_train_batch, [len(x_train_batch), 128, 1, 9])

                self.sess.run(self.optim, feed_dict={self.x: x_train_batch, self.y: y_train_batch})
                acc_train, loss_train = self.sess.run([self.accuracy, self.loss],
                                         feed_dict={self.x: x_train_batch, self.y: y_train_batch})
                # acc_val, loss_val = self.sess.run([self.accuracy, self.loss],
                #                          feed_dict={self.x: np.reshape(self.X_val,[len(self.X_val),128,1,9]), self.y: np.reshape(self.Y_val,[len(self.Y_val),6])})

                # _, acc_train, loss_train = self.sess.run([self.metrics(self.y,self.prediction)[2], self.accuracy, self.loss], feed_dict={self.x: x_train_batch, self.y: y_train_batch})
                # _, acc_val, loss_val = self.sess.run([self.metrics(self.y,self.prediction)[2], self.accuracy, self.loss], feed_dict={self.x: np.reshape(self.X_val,[len(self.X_val),128,1,9]), self.y: np.reshape(self.Y_val,[len(self.Y_val),6])})



            
            if epoch%display_step ==0:
                print("cost after epoch %i :  %.3f" % (epoch + 1, loss_train), end="")
                print("  train accuracy   :  %.3f" % acc_train)
                

            # if epoch%validation_step ==0:
                
            #     # val_optimizer, val_loss, val_prediction, val_accuracy  = self.sess.run([self.optim, self.loss, self.prediction, self.accuracy], feed_dict = {self.x: x_val, self.y: y_val})
            #     print(f'epoch: {epoch} val accuracy: {acc_val} loss: {loss_val}')
            #     # print("  val accuracy   :  %.3f" % (acc_val))
               
                

            if epoch%checkpoint_step ==0:
                self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
                if os.path.isfile(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'))== False:
                    
                    os.makedirs(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'),exist_ok=True)
                    self.saver.save(self.sess, os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints','my-model'))
                    #,global_step=1
                else:
                    self.saver.save(self.sess, os.path.join(os.getcwd(), self.base_dir, self.model_name, 'chekpoints','my-model'))


            if epoch%summary_step ==0:
               
                if os.path.isfile(os.path.join(os.getcwd(),self.base_dir,self.model_name, 'summaries'))== False:
                    os.makedirs(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'),exist_ok=True)
                    tf.summary.FileWriter(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'), self.sess.graph)
                else:
                    tf.summary.FileWriter(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'), self.sess.graph)

          
        print("network trained")
        

    def test_model(self):
        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = self.load_data()
        # self.load()
        # predictions, acc_final, loss_final = self.sess.run([self.metrics(self.y,self.prediction)[2], self.accuracy, self.loss], feed_dict={self.x : self.X_test , self.y: self.y_test})
        # print(f'final results: accuracy: {acc_final} loss: {loss_final}')
        # _, acc_test, loss_test = self.sess.run([self.metrics(self.y,self.prediction)[2], self.accuracy, self.loss], feed_dict={self.x: self.X_test, self.y: self.y_test})
        
        test_accuracy = self.sess.run(self.accuracy, feed_dict={self.x: np.reshape(self.X_test,[len(self.X_test),128,1,9]), self.y: np.reshape(self.Y_test,[len(self.Y_test),6])})
        print("  test accuracy   :  %.3f" % (test_accuracy))





    @abstractmethod
    def network(self, X):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self, Y, y_pred):
        raise NotImplementedError('subclasses must override metrics()!')


