from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)

from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten,
                          GlobalMaxPool2D, MaxPool2D, AveragePooling2D,
                          concatenate, Activation, Concatenate, Input,
                         Dropout, Dense, Add)
from keras.utils import Sequence, to_categorical
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import kaggle_util
import numpy as np
from audio_1d import get_1d_conv_head
from seresnet import *
from keras.applications.xception import Xception


def conv2d_b(xin, filters, kernel_size, padding='same', strides=1):
    x = Convolution2D(filters, 
                      kernel_size = kernel_size, 
                      padding = padding, 
                      strides = strides,
                      kernel_initializer='he_normal')(xin)
    x = BatchNormalization()(x)
    return x

def conv2d_bn(x, filters, kernel_size, padding='same', strides=1):
    x = conv2d_b(x, filters, kernel_size, padding, strides)
    x = LeakyReLU(0.1)(x)
    return x

def mixup_cross_entropy_loss(y_true, y_pre):
    #assert(y_true.shape == y_pre.shape)
    input = K.log(K.clip(K.softmax(y_pre), 1e-5, 1))
    loss = - K.mean(input * y_true)
    return loss

def make_layers(x, cfg, batch_norm=False):
    for v in cfg:
        if v == 'M':
            x = MaxPool2D((2,2), strides=2)(x)
        else:
            x = Convolution2D(v, kernel_size = 3, padding='same')(x)
            if batch_norm:
                x = BatchNormalization()(x)
            x = LeakyReLU(0.1)(x)
    return x
def dense_layers(x, cfg_dense, flatten = True):
    if flatten:
        x = Flatten()(x)
    
    for (v, drop) in cfg_dense:
        x = Dense(v, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        if drop > 0:
            x = Dropout(rate=drop)(x)
    return x

def dense_outlayer(x, nclass, cfg_dense, flatten = True):
    x = dense_layers(x, cfg_dense, flatten)
    out = Dense(nclass, activation=softmax)(x)
    
    return out

def make_model(config, inp, out, opt='adam'):
    model = models.Model(inputs=inp, outputs=out)
    if opt == 'adam':
        opt = optimizers.Adam(config.learning_rate)
    elif opt == 'sgd':
        opt = 'sgd'
        print('sgd opt')
    loss = losses.categorical_crossentropy
    model.compile(optimizer=opt, loss=loss, metrics=['acc'])
    #model.summary()
    return model

def make_vgg(config, cfg, length, batch_norm = True, dense_cfg = [(512,0.2), (1024,0.2)]):
    print('configs', cfg)
    
    nclass = config.n_classes
    
    inp = Input(shape=(config.n_mels, length, 1))
    x = make_layers(inp, cfg, batch_norm)

    #maxpool = GlobalMaxPool2D()(x)
    #avgpool = GlobalAveragePooling2D()(x)
    #x = concatenate([maxpool, avgpool])
    #x = Dropout(rate=0.5)(x)
    #x = maxpool
    out = dense_outlayer(x, nclass, dense_cfg, True)
    
    return make_model(config, inp, out)

def make_numeric(config, cfg, length, dense_cfg):
    nclass = config.n_classes
    inp = Input(shape=(length, 1))
    #x = make_depp_layers(inp, cfg)
    
    #x = Flatten()(inp)
    out = dense_outlayer(inp, nclass, dense_cfg, True)
    return make_model(config, inp, out)

def make_seresnet(mname, config, length, dense_cfg = [(512,0.2), (1024,0.2)]):
    if 'seresnet18' in mname:
        print('seresnet18')
        model = SEResNet50(input_shape=(config.n_mels, length, 1),
                          bottleneck=True,
                         classes = config.n_classes)
    elif 'seresnet34' in mname:
        print('seresnet34')
        model = SEResNet50(input_shape=(config.n_mels, length, 1),
                          bottleneck=True,
                         classes = config.n_classes)
    elif 'seresnet50' in mname:
        print('seresnet50')
        model = SEResNet50(input_shape=(config.n_mels, length, 1),
                          bottleneck=True,
                         classes = config.n_classes)
    opt = optimizers.Adam(config.learning_rate)
    loss = losses.categorical_crossentropy
    model.compile(optimizer=opt, loss=loss, metrics=['acc'])
    
    return model

def make_depp_layers(x, cfg, batch_norm=True):
    for (v, drop) in cfg:
        x = Dense(v)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        if drop > 0:
            x = Dropout(rate=drop)(x)
    return x

def block_residual(x, filters, kernel_size = 3, strides = 1, block = 1):
    #print(x.shape, block, filters, kernel_size)
    y1 = conv2d_bn(x, filters, kernel_size, padding = 'same', strides=strides)
    #print('y1', y1.shape)
    y2 = conv2d_b(y1, filters, kernel_size, padding = 'same', strides=1)
    #print('y2', y2.shape)
    
    if block == 0:
        padding = 'valid' if block != 0 else 'same'
        shortcut = conv2d_b(x, filters, kernel_size, padding = padding, strides=strides)
    else:
        shortcut = x

    #print(y2.shape, shortcut.shape)    
    y = Add()([y2, shortcut])
    y = LeakyReLU(0.1)(y)
    
    return y

def make_2rdcnn(config, cfg, length, dense_cfg):
    nclass = config.n_classes
    inp = Input(shape=(config.n_mels, length, 1))
    x = conv2d_bn(inp, 64, (7,3))
    x = MaxPool2D((1,3))(x)
    x = conv2d_bn(x, 128, (1,7))
    x = MaxPool2D((1,4))(x)
    x = conv2d_bn(x, 256, (1,10))
    x = conv2d_bn(x, 256, (1,10))
    x = conv2d_bn(x, 512, (7,1))
    x = conv2d_bn(x, 512, (7,1))
    x = GlobalMaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    out = dense_outlayer(x, nclass, dense_cfg, False)
    return make_model(config, inp, out)

def make_reslayer(x, cnt, filters, kernel_size, strides=1):
    for i in range(cnt):
        x = block_residual(x, filters, kernel_size, strides, i)
    return x

def make_resnet_layers(inp, cfg):
    x = conv2d_bn(inp, 64, 7, strides=2)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    
    x = make_reslayer(inp, cfg[0], 64, 3, 1)
    x = make_reslayer(x, cfg[1], 128, 3, 2)
    x = make_reslayer(x, cfg[2], 256, 3, 2)
    x = make_reslayer(x, cfg[3], 512, 3, 2)
    x = AveragePooling2D(pool_size=1, strides=1)(x)
    return x

def make_resnet(config, cfg, length):
    nclass = config.n_classes
    inp = Input(shape=(config.n_mels, length, 1))
    
    x = make_resnet_layers(inp, cfg)
    out = dense_outlayer(x, nclass)
    return make_model(config, inp, out)

def ResNeXtBottleneck(x, filters, strides, cardinality, base_width, widen_factor, block=1):
    width_ratio = filters / (widen_factor * 64.)
    D = cardinality * int(base_width * width_ratio)
    
    y1 = conv2d_bn(x, D, 1, padding='valid', strides=1)
    y2 = conv2d_bn(y1, D, 3, padding='same', strides=strides)
    y3 = conv2d_b(y2, D, 1, padding='valid', strides=1)
    #print(block, filters, D)
    if block == 0:
        padding = 'valid' if block != 0 else 'same'
        shortcut = conv2d_b(x, D, 1, padding = padding, strides=strides)
    else:
        shortcut = x
        
    y = Add()([y3, shortcut])
    y = LeakyReLU(0.1)(y)
    
    return y

def ResNextBlock(x, filters, block_depth, pool_stride, cardinality, 
                 base_width, widen_factor):
    block = 0
    for bottlenect in range(block_depth):
        stride = pool_stride if bottlenect == 0 else 1
        x = ResNeXtBottleneck(x, filters, stride, cardinality, 
                                  base_width, widen_factor, block)
        block += 1
        
    return x
    
def make_resnext_layers(x, cardinality = 8, depth = 29, base_width = 64, widen_factor = 4):
    block_depth = (depth - 2) // 9

    x = conv2d_bn(x, 64, 3, padding='same', strides=1)
    x = ResNextBlock(x, 64 * widen_factor, block_depth, 1, cardinality, base_width, widen_factor)
    x = ResNextBlock(x, 128 * widen_factor, block_depth, 2, cardinality, base_width, widen_factor)
    x = ResNextBlock(x, 256 * widen_factor, block_depth, 2, cardinality, base_width, widen_factor)
    x = AveragePooling2D(pool_size=8, strides=1)(x)
    
    return x

def make_resnext(config, length, cardinality = 1, depth = 29, base_width = 32, widen_factor = 2):
    nclass = config.n_classes
    inp = Input(shape=(config.n_mels, length, 1))
    
    x = make_resnext_layers(inp, cardinality, depth, base_width, widen_factor)  
    out = dense_outlayer(x, nclass)
    return make_model(config, inp, out)
    
def make_mixmodel(config, multi_cfg, final_dense):
    nclass = config.n_classes
    inputs = []
    model_outpus = []
    for (name, width, length, cfg, cfg_dense) in multi_cfg:
        print(name, cfg_dense)
        if 'cnn1d' == name:
            inp = Input(shape=(config.audio_length,1), name=name)
            x = get_1d_conv_head(inp, False)
        elif 'numeric' == name:
            inp = Input(shape=(length, 1), name=name)
            x = inp
            #x = make_depp_layers(inp, cfg[1])
        else:
            inp = Input(shape=(width, length, 1), name=name)
            if 'vgg' in cfg[0]:
                x = make_layers(inp, cfg[1], True)
            elif 'resnet' in cfg[0]:
                x = make_resnet_layers(inp, cfg[1])
            elif 'resnext' in cfg[0]:
                x = make_resnext_layers(inp)
            elif 'pre_xception' in cfg[0]:
                backend = Xception(weights='imagenet', include_top=False, input_shape = (128, 157, 3))
                for i in range(len(backend.layers)):
                   backend.layers[i].name = '{}_{}'.format(name, backend.layers[i].name)
                   #print(i, backend.layers[i].name)

                x = backend.output
                inp = backend.input
            #maxpool = GlobalMaxPool2D()(x)
            #avgpool = GlobalAveragePooling2D()(x)
            #x = concatenate([maxpool, avgpool])
            #x = Flatten()(x)
            
        #x = Dropout(rate=0.1)(x)
        inputs.append(inp)
        
        x = dense_layers(x, cfg_dense, 'cnn1d' != name)
        
        #x = Dense(neros)(x)
        #x = BatchNormalization()(x)
        #x = LeakyReLU(0.1)(x)
        #x = Dropout(rate=0.1)(x)
        model_outpus.append(x)
    
    print(inputs)
    x = concatenate(model_outpus)
    x = dense_layers(x, final_dense, False)
    
    """
    main_l = BatchNormalization()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(rate=0.5)(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(rate=0.2)(x)
    """
    
    out = Dense(nclass, activation=softmax)(x)
    return make_model(config, inputs, out)
