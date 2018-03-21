"""
Usage:
1. training
python run_this.py --action='train'\
    -p ./train -v ./val
2. prediction
python run_this.py --action='predict'\
    -p /test/**.jpg
"""
import time
import json
import argparse
import sys
sys.path.append('..')
from model.mobilenet import MobileNet 
import numpy as np
import cv2
import os
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard,SGDLearningRateTracker
from keras.models import load_model


def augmean(img,imgNorm,width,height):
    if imgNorm == "sub_and_divide":
            img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
    elif imgNorm == "sub_mean":
            img = cv2.resize(img, ( width , height ))
            img = img.astype(np.float32)
            img[:,:,0] -= 103.939
            img[:,:,1] -= 116.779
            img[:,:,2] -= 123.68
    elif imgNorm == "divide":
            #img = cv2.resize(img, ( width , height ))
            img = img.astype(np.float32)
            img = img/255.0
            img -= 0.5
            img *= 2.
    return img

def preprocess_input_aug(x):
    x=augmean(x,"sub_mean",width=224,height=224)
    return x
def preprocess_input(x):
    x=augmean(x,"sub_mean",width=224,height=224)
    return x

def parse_json(fname):
    """Parse the input profile

    @param fname: input profile path

    @return data: a dictionary with user-defined data for training

    """
    with open(fname) as data_file:
        data = json.load(data_file)
    return data

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def write_json(data, fname='./output.json'):
    """Write data to json

    @param data: object to be written

    Keyword arguments:
    fname  -- output filename (default './output.json')

    """
    with open(fname, 'w') as fp:
        json.dump(data, fp, cls=NumpyAwareJSONEncoder)


def print_time(t0, s):
    """Print how much time has been spent

    @param t0: previous timestamp
    @param s: description of this step

    """

    print("%.5f seconds to %s" % ((time.time() - t0), s))
    return time.time()


def main():
    parser = argparse.ArgumentParser(
        description="MobileNet example."
        )
    parser.add_argument(
        "--batch-size", type=int, default=32, dest='batchsize',
        help="Size of the mini batch. Default: 32."
        )
    parser.add_argument(
        "--action", type=str, default='train',
        help="Action to be performed, train/predict"
        )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of epochs, default 20."
        )
    parser.add_argument(
        "--lr", type=float, default=0.01,                                                                                                                                                                                      
        help="Learning rate of SGD, default 0.001."
        )
    parser.add_argument(
        "--epsilon", type=float, default=1e-8,
        help="Epsilon of Adam epsilon, default 1e-8."
        )
    parser.add_argument(
        "-p", "--path", type=str, default='./train', dest='trainpath',#required=True,
        help="Path where the images are. Default: $PWD."
        )
    parser.add_argument(
        "-v", "--val-path", type=str,default='./val',
        dest='valpath', help="Path where the val images are. Default: $PWD."
        )
    parser.add_argument(
        "--img-width", type=int, default=224, dest='width',
        help="Rows of the images, default: 224."
        )
    parser.add_argument(
        "--img-height", type=int, default=224, dest='height',
        help="Columns of the images, default: 224."
        )
    parser.add_argument(
        "--channels", type=int, default=3,
        help="Channels of the images, default: 3."
        )

    args = parser.parse_args()
    sgd = SGD(lr=args.lr, decay=0.0001,momentum=0.9)#,
    #adma=Adam(args.lr,beta_1=0.9,beta_2=0.999,epsilon=1e-8)

    t0 = time.time()
    if args.action == 'train':
        train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input_aug,
                                        #rescale=1./255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)       
        validation_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
                                            #rescale=1./255)
        train_generator=train_datagen.flow_from_directory(args.trainpath,target_size=(224, 224),batch_size=args.batchsize)
        validation_generator=validation_datagen.flow_from_directory(args.valpath,target_size=(224, 224),batch_size=args.batchsize)
        classes = train_generator.class_indices
        nb_train_samples = train_generator.samples
        nb_val_samples = validation_generator.samples
        print("[demo] N training samples: %i " % nb_train_samples)
        print("[demo] N validation samples: %i " % nb_val_samples)
        nb_class = train_generator.num_classes
        print('[demo] Total classes are %i' % nb_class)

        t0 = print_time(t0, 'initialize data')
        model = MobileNet(input_shape=(args.height, args.width,args.channels),alpha=0.25,classes=nb_class)
        # dp.visualize_model(model)
        t0 = print_time(t0, 'build the model')

        model.compile(
            #optimizer=sgd, loss='categorical_crossentropy',
            optimizer=sgd, loss='binary_crossentropy',
            metrics=['accuracy'])
        t0 = print_time(t0, 'compile model')
        tb_cb = TensorBoard(log_dir='./logs', 
                            write_images=True, histogram_freq=0, write_graph=True)
        model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples//args.batchsize,
            epochs=args.epochs,
            validation_data=validation_generator,
            validation_steps=nb_val_samples//args.batchsize,
            callbacks=[tb_cb,SGDLearningRateTracker('./learning.txt')])
        t0 = print_time(t0, 'train model')

        model.save_weights('./weights/weights.h5', overwrite=True)
        model_parms = {'nb_class': nb_class,
                       'nb_train_samples': nb_train_samples,
                       'nb_val_samples': nb_val_samples,
                       'classes': classes,
                       'channels': args.channels,
                       'height': args.height,
                       'width': args.width}
        write_json(model_parms, fname='./logs/model_parms.json')
        t0 = print_time(t0, 'save model')
    elif args.action == 'predict':
        _parms = parse_json('./logs/model_parms.json')
        model = MobileNet(input_shape=(args.height, args.width,args.channels),alpha=0.25,classes=2)
        weights_path='./weights/weights.h5'
        model.load_weights(weights_path)
        model.compile(
            optimizer=sgd, loss='binary_crossentropy',
            metrics=['accuracy'])
        t0 = print_time(t0, 'prepare data')
        subdirs=os.listdir(args.valpath)
        
        imagenames=[]
        imagesarray=[]
        m_class=[]
        for subdir in subdirs:
            images=os.listdir(args.valpath+'/'+str(subdir))
            for image in images:
                img=cv2.imread(args.valpath+'/'+subdir+'/'+image,cv2.COLOR_BGR2RGB)
                img=cv2.resize(img,(224,224))
                imagenames.append(image)
                m_class.append(subdir)
                img=preprocess_input(img)
                imagesarray.append(img)
        classes = _parms['classes']
        accpreds=0
        for i,m_c in zip(range(len(imagenames)),m_class):
            cruentimg=imagesarray[i][np.newaxis,:,:,:]
            print(cruentimg.shape)
            results = model.predict(imagesarray[i][np.newaxis,:,:,:])
            _cls=np.argmax(results,axis=1)
            print(m_c)
            print(_cls[0])
            if(_cls[0]==int(m_c)):
                accpreds=accpreds+1
            max_prob = results[0][_cls]
            print('[demo] %s: %s (%.2f)' % (str(imagenames[i]), str(_cls), max_prob))
        print(accpreds)
        print(accpreds/len(m_class))
        print("total acc :(%.3f)", float(accpreds/len(m_class)))
        t0 = print_time(t0, 'predict')

if __name__ == '__main__':
    main()
