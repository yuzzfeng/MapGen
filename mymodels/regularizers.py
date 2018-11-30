from keras import backend as K
from keras.regularizers import Regularizer

##Simple implementation
#def linf_reg(weight_matrix):
#    return K.max(K.abs(weight_matrix))
    
class linf_reg(Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        output = self.layer.get_output(True)
        return K.sum(K.max(K.abs(output), axis=0))
        
        #loss += self.l1 * K.sum(K.mean(K.abs(output), axis=0))
        #loss += self.l2 * K.sum(K.mean(K.square(output), axis=0))
        #return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2}
    
class linf_reg_weight(Regularizer):
        """Regularizer for L1 and L2 regularization.
        # Arguments
            l1: Float; L1 regularization factor.
            l2: Float; L2 regularization factor.
        """

        def __init__(self, l1=0., l2=0.):
            self.l1 = K.cast_to_floatx(l1)
            self.l2 = K.cast_to_floatx(l2)

        def __call__(self, x):
            return K.max(K.abs(x))

        def get_config(self):
            return {'l1': float(self.l1),
                    'l2': float(self.l2)}