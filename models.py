from keras import layers, Model
from keras import regularizers
from keras.activations import softmax
import keras.backend as K
import numpy as np
import tensorflow as tf

def soft_with_T(T=1):
    def swt(x):
        return softmax(x/T)
    return swt

def exp_regularizer(b, l2=0.01):
    l = np.logspace(1, b, num=10, base=np.exp(1))
    e_x = np.exp(l - np.max(l))
    s = e_x / e_x.sum(axis=0)
    var_s = K.variable(value=s)
    def exp_reg(x):
        x_sorted = tf.sort(x, axis=-1, direction='ASCENDING', name=None)
        return l2 * K.sqrt(K.sum((x_sorted-s)**2))
    return exp_reg
    
class TeacherModel(Model):
    def __init__(self, input_shape, num_classes):
        self.x_in = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(self.x_in)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.50)(x)
        #x = layers.Flatten()(self.x_in)
        #x = layers.Dense(1200, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        #x = layers.Dropout(0.25)(x)
        #x = layers.Dense(1200, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        #x = layers.Dropout(0.25)(x)
        self.logit = layers.Dense(num_classes, activation='linear')(x)
        self.out = layers.Activation('softmax')(self.logit)
        super().__init__(self.x_in, self.out)
    
    def T_model(self, T):
        out = layers.Activation(soft_with_T(T))(self.logit)
        return Model(self.x_in, out)
        
class SoftTeacherModel(Model):
    def __init__(self, input_shape, num_classes, l1=0, l2=0, b=2):
        x_in = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x_in)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.50)(x)
#         out = layers.Dense(num_classes, activation='softmax', activity_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
        out = layers.Dense(num_classes, activation='softmax', activity_regularizer=exp_regularizer(b, l2=l2))(x)
        super().__init__(x_in, out)
        
class StudentModel(Model):
    def __init__(self, input_shape, num_classes, softmax_l=0, T=1, l2=0.1, b=2, in_class=False):
        x_in = layers.Input(shape=input_shape)
#         x = layers.Conv2D(2, kernel_size=(3, 3), activation='relu')(x_in)
#         x = layers.Conv2D(2, kernel_size=(3, 3), activation='relu')(x)
#         x = layers.MaxPooling2D()(x)
#         #x = layers.Dropout(0.25)(x)
#         x = layers.Flatten()(x)
#         x = layers.Dense(4, activation='relu')(x)
        #x = layers.Dropout(0.50)(x)
        #out_1 = layers.Dense(num_classes, activation='softmax', activity_regularizer=exp_regularizer(b, l2=l2), name='o1')(x)
        #out_1 = layers.Activation(soft_with_T(T), name='o1')(x)
        #out_1 = layers.ActivityRegularization(l2=softmax_l, name='o1')(out_1)
        #out_2 = layers.Activation('softmax', name='o2')(x)
        
        x = layers.Flatten()(x_in)
        x = layers.Dense(800, activation='relu')(x)
        x = layers.Dense(800, activation='relu')(x)
        
        s = layers.Dense(num_classes, activation='linear')(x)
        out_1 = layers.Activation(soft_with_T(T), name='o1')(s)
        out_2 = layers.Activation('softmax', name='o2')(s)
#         out_1 = layers.Lambda(lambda x:x, name='o1')(s)
#         out_2 = layers.Lambda(lambda x:x, name='o2')(s)
        if in_class:
            super().__init__(x_in, [out_1, out_2])
        else:
            super().__init__(x_in, out_2)
            

class StudentModelDense(Model):
    def __init__(self, input_shape, num_classes, softmax_l=0, in_class=False):
        x_in = layers.Input(shape=input_shape)
        x = layers.Flatten()(x_in)
        x = layers.Dense(5, activation='relu')(x)
        x = layers.Dense(num_classes, activation='linear')(x)
        out_1 = layers.Activation('softmax', activity_regularizer=regularizers.l2(softmax_l), name='o1')(x)
        out_2 = layers.Activation('softmax', name='o2')(x)
        if in_class:
            super().__init__(x_in, [out_1, out_2])
        else:
            super().__init__(x_in, out_1)
        