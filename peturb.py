import tensorflow as tf
from model import create_ds_cnn_model
from utils import model_info, calculate_mfcc
from tensorflow.keras.models import load_model


#Creating adversarial peturbation 
def create_peturbation_from_mfcc(audio, label):

    #Set Model
    model = load_model('saved_model\ds_cnn_pure.h5')

    #Set Peripherals
    loss_object = tf.keras.losses.CategoricalCrossentropy()

    with tf.GradientTape() as tape:
        tape.watch(audio)         #Watch Audio to get gradient

        #load MFCC
        mfcc = calculate_mfcc(audio)
        mfcc = tf.reshape(mfcc, [-1])    
        mfcc = tf.reshape(mfcc, (1, 490))
        
        prediction = model(mfcc)  #Predict with its MFCC
        loss = loss_object([label], prediction)

    gradient = tape.gradient(loss, audio)
    # print(gradient)
    signed_grad = tf.sign(gradient)
    return signed_grad


