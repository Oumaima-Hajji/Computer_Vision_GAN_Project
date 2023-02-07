
from matplotlib import image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from pickletools import optimize
from pyexpat import model
from sklearn import metrics
import time
import numpy as np
import csv
from skimage import color
from skimage import io
import PIL
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np  # for using np arrays
from tensorflow.keras.layers import *
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
import metrics as metrics
from keras.models import Sequential
import tensorflow.keras as keras
from tensorflow.keras import layers
from keras.layers import Activation, Dense
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Reshape
from keras.layers import Input
from sklearn.metrics import jaccard_score
import math as math
from keras import callbacks
from keras.callbacks import EarlyStopping
from keras.layers import Lambda
import lossaleatoire as LossAleatory
import imageprocessing as GestionImage
import shutil
from keras.models import Model
from sklearn.model_selection import train_test_split

# opt = tf.keras.optimizers.Adam(learning_rate=0.001)

import time
tf.compat.v1.disable_eager_execution()


class Gan():
    def __init__(self, denominator):
        '''
        The constructor where we initialize the data needed and called in by other methods
        '''

        self.input_size = (250, 250, 1)

        self.n_classes = 3
        self.latent_dim = 100
        self.grey = True

        self.data = {"num_cas": 0,
                     "list_cas": [["mean_squared_error", "binary_cross_entropy"], ["mean_squared_error", "wasserstein"], ["mean_squared_error", "L1"], ["mean_squared_error", "L2"], ["mean_squared_error", "wasserstein"], ["mean_squared_error", "perceptual_wasserstein"],
                                  ["binary_cross_entropy", "categorical_cross_entropy"], ["binary_cross_entropy", "L1"], ["binary_cross_entropy", "L2"], [
                                      "binary_cross_entropy", "perceptual_wasserstein"], ["binary_cross_entropy", "categorical_cross_entropy"],
                                  ["wasserstein", "L1"], ["wasserstein", "L2"], [
                                      "wasserstein", "perceptual_wasserstein"], ["wasserstein", "categorical_cross_entropy"],
                                  ["L2", "L1"], ["L2", "categorical_cross_entropy"], [
                                      "L2", "perceptual_wasserstein"],
                                  ["L1", "categorical_cross_entropy"], [
                         "L1", "perceptual_wasserstein"],
                         ["perceptual_wasserstein", "categorical_cross_entropy"]],
                     "cas": [0, 1],
                     # à remplacer plus tard par une liste pour le dénominateur dans le main
                     "denominator": denominator,
                     "total_epoch": 100,
                     "epoch": 0,
                     "nom_chemain": [],
                     "y_true": "true",
                     "y_pred": "pred"
                     }
        self.channels = 1

    # Discriminator PART

    def discriminator(self):
        '''
        This function represents the discriminator part of the GAN
        It returns a Model
        '''

        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="relu"))  # sigmoid
        model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

    # Generator PART

    def sampling(self, mu_log_variance):
        '''
        sample from a normal distribution
        arguments are [mu, log_var], where log_var is the log of the squared sigma
        '''
        mu, log_variance = mu_log_variance
        # Returns a tensor with normal distribution of values.
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mu), mean=0.0, stddev=1.0)

        random_sample = mu + \
            keras.activations.exponential(log_variance/2) * epsilon
        return random_sample

    # Encoder

    def build_encoder(self):
        '''
        This function is where we build the Encoder part of the GAN
        It returns a Model
        '''

        img = Input(shape=self.input_size)

        h = Flatten()(img)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        print("SHAPE :", h.shape)  # 512
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        latent_repr = Lambda(
            self.sampling, name="encoder_output")([mu, log_var])
        return Model(img, latent_repr)

    def build_decoder(self):
        '''
        This function is where we build the Decoder part of the GAN
        It returns the model
        '''

        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.prod(self.input_size), activation='tanh'))
        model.add(Reshape(self.input_size))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        img = model(z)
        return Model(z, img)

    def renameimage(self, path):
        '''
        This function is used to label the images and give them proper names depending on their origin.
        There are 2 type of images : image data that represent pathologies, labeled as "Q352" and data that represent physiologies, labeled as WT

        The labeled data is then saved into a new folder called "dataset_rename"
        '''
        count_patho = 0
        count_physio = 0
        self.path = path

        for file in os.listdir(self.path):
            print("FILE IS :", file)
            if file != ".DStore":
                # for filename in file:
                full_name_file = self.path + file
                new_name = "os.getcwd() + dataset_rename/"
                if "Q352" in file:
                    print("here")
                    new = new_name + str(count_patho) + "-Q352.jpg"
                    count_patho = count_patho + 1
                if "WT" in file:
                    print("here 2")
                    new = new_name + str(count_physio) + "-WT.jpg"
                    count_physio = count_physio + 1
                print("new :", new)
                cmd = "cp " + full_name_file + " " + new
                print("cmd :", cmd)
                try:
                    # os.system(cmd)
                    shutil.copy(full_name_file, new)
                except Exception as e:
                    print("Error is : ", e)

    def compile(self):
        '''
        This function compiles the discriminator and encoder of the GAN with the LOSS functions.
        '''
        # compile the Discriminator:
        # Call train and test data
        data_train = GestionImage.GestionImage()
        X_train, X_test,  y_train, y_test = data_train.splitdata()

        optimizer = keras.optimizers.Adam(learning_rate=0.0002)  # 0.01
        self.model_disc = self.discriminator()
        print("SELF LOSS IS :", self.loss_)

        loss = self.loss_
        self.model_disc.compile(
            loss=loss, optimizer=optimizer, metrics=['accuracy'])
        # loss = self.loss_  , optimizer= keras.optimizers.Adam(learning_rate=0.01),  metrics=['accuracy', keras.metrics.MeanIoU(num_classes=2)]) #, jaccard_score(x_true[0],y_pred[0])])
        self.model_disc.summary()

        # Compile the Encoder :

        self.encod = self.build_encoder()
        self.model_decod = self.build_decoder()
        img = Input(shape=self.input_size)
        valid = self.encod(img)
        sortie = self.model_decod(valid)

        self.model_disc.trainable = False
        validity = self.model_disc(valid)

        self.autoenc_model = Model(img, [sortie, validity])

        self.autoenc_model.compile(loss=['mse', 'binary_crossentropy'],
                                   loss_weights=[0.999, 0.001],
                                   optimizer=optimizer)
        # ( loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer= keras.optimizers.Adam(learning_rate=0.01),  metrics=['accuracy', keras.metrics.MeanIoU(num_classes=2)]) #, jaccard_score(x_true[0],y_pred[0])])
        self.autoenc_model.summary()

    def write_csv(self, name, list):
        '''
        This function writes rows in a csv file
        '''
        # open the file in the write mode
        with open(name, 'a') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row in the csv file
            writer.writerow(list)

    def recuperer_images(self, epoch, name_loss, num_cas, d_loss, denominator, path, imgs):
        '''
        This function takes args that are related to the operation made 
        like the case number, the number of epoch, the loss functions used, 
        the path of the generated picture and the picture itself in one CSV file
        In order to organize the results and compare

        '''

        r, c = 32, 40  # 40 generated images and 32 images in the test/train
        z = np.random.normal(size=(r*c, self.latent_dim))
        gen_imgs = self.model_decod.predict(z)  # decoder prediction
        gen_imgs = 0.5 * gen_imgs + 0.5
        headers = ["original", "generated", "num_epoch", "name_loss", "loss",
                   "list_cas", 'numcas', "denominator",  "PSNR score", "SSIM score"]
        self.write_csv("bilan.csv", headers)
        i = 0
        for j in range(c):
            imaged = gen_imgs[j]
            if self.grey == False:
                I = imaged
                print("NOT THERE")
            if self.grey == True:
                print("There")
                I = imaged[:, :, 0]
            name = path + "_" + str(i) + "_" + str(j) + "_" + str(epoch) + \
                "_" + str(name_loss) + "_" + str(denominator) + ".png"
            plt.imsave(name, I, cmap="gray")
            metrica = metrics.Metric()
            for i in range(r):
                originale = imgs[i]
                print("imgs :", imgs.shape)
                scores = metrica.compare_images(originale, imaged)
                self.write_csv("bilan.csv", [originale, imaged, epoch, name_loss, d_loss,
                               self.data["list_cas"], self.data["num_cas"], denominator, scores[0], scores[1]])


############################ Training #################################################

    def train(self, epochs, batch_size=10, sample_interval=50):
        imgg = self.build_decoder()
        print("decoder-image:", imgg)

        # start counting time
        headers = ["execution time"]
        self.write_csv("temps_execution.csv", headers)
        start_time = time.time()
        print("start time of training :", start_time)

        data_train = GestionImage.GestionImage()
        data, label = data_train.load_data()
        # Converting the list of array into numpy array
        data = np.asarray(data)
        label = np.asarray(label)
        # Split train and test with train_test_split defautlt:
        X_train, X_test, y_train, y_test = train_test_split(
            data, label, test_size=0.33)
        # Configure input  Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        print("SHAPE :", X_train.shape)
        # Need to add a fourth dimension if the library didn't add it... :
        X_train = X_train.reshape(
            X_train.shape[0], X_train.shape[1], X_train.shape[2], self.channels)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        new_loss_list = []
        headers = ["epoch", "dloss", "acc", "gloss", "mse"]
        self.write_csv("liste_des_loss.csv", headers)
        # self.save_headers_xlsx(headers, self.name_sheet_xlsx)
        idx = 0

        for epoch in range(epochs):
            self.data["epoch"] = epoch

            # vérifier quel modèleprendre en changeant la loss :
            la = LossAleatory.LossAleatory(self.data)
            self.loss_ = la.randomization()
            # la.self.loss()
            if self.loss_ == "mean_squared_error":
                self.loss_ = la.mean_squared_error
            if self.loss_ == "wasserstein":
                self.loss_ = la.wasserstein
            if self.loss_ == "binary_cross_entropy":
                self.loss_ = "binary_crossentropy"
            if self.loss_ == "L1":
                self.loss_ = la.L1
            if self.loss_ == "L2":
                self.loss_ = la.L2
            if self.loss_ == "perceptual_wasserstein":
                self.loss_ = la.perceptual_wasserstein
            if self.loss_ == "categorical_cross_entropy":
                self.loss_ = la.categorical_cross_entropy

            self.compile()
            # ---------------------
            # Train Discriminator
            # ---------------------
            # Select a random batch of images
            idx = np.random.randint(0,  X_train.shape[0], batch_size)
            print("INDEW :", idx)
            imgs = X_train[idx]
            ("SHAPE ===============", imgs.shape)
            # np.expand_dims(np.squeeze(imgs), axis=3))# erreur de dimension sinon 10, 250, 250, 1, 1 alors qu'onveut n= 4
            latent_fake = self.encod.predict(imgs)
            self.data["y_pred"] = latent_fake
            latent_real = np.random.normal(size=(batch_size, self.latent_dim))
            self.data["y_true"] = latent_real

            # Train the discriminator
            d_loss_real = self.model_disc.train_on_batch(latent_real, valid)
            d_loss_fake = self.model_disc.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            # Train Generator
            # --------------------
            # Train the generator
            g_loss = self.autoenc_model.train_on_batch(imgs, [imgs, valid])

            # Plot the progress
            print("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" %
                  (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
            new_loss_list = [epoch, d_loss[0],
                             100*d_loss[1], g_loss[0], g_loss[1]]
            # print("SAVING", self.name_sheet_xlsx)
            # self.save_loss_descriptors(self.name_sheet_xlsx, new_loss_list, str(epoch+2))
            self.write_csv("liste_des_loss.csv", new_loss_list)
            print("new_loss_list:", new_loss_list)
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                name_loss = la.randomization()
                self.recuperer_images(epoch, name_loss, self.data["num_cas"], d_loss[0], self.data["denominator"],
                                      "write_file_path_here", imgs)


            # stop counting time
            finish_time = time.time() - start_time
            print("--- %s seconds ---" % (finish_time))
            self.write_csv("temps_execution.csv",  [
                           epoch, name_loss, self.data["num_cas"], finish_time])


############################ Plotting #################################################


    def plot_loss(self, epochloss_file):

        reader = csv.reader(epochloss_file)
        epochs = []
        loss_train = []
        for row in reader:
            for i in row:
                print("first epoch:", row[0])
                print("first loss:", row[1])
                epochs = epochs.append(row[0])
                loss_train = loss_train.append(row[1])
                plt.plot(epochs, loss_train, 'g', label='Trainingloss')
                # plt.plot(epochs, loss_val, 'b', label='validation loss')
                plt.title('Training loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()


if __name__ == "__main__":

    class_ = Gan(2) #this changes
    class_.renameimage(os.getcwd + "dataset_resize/")
    class_.train(epochs=501, batch_size=32, sample_interval=50)
    class_.compile()
