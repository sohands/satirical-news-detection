from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomNormal

class AttentionLayer(Layer):
    def __init__(self, hidden_layers=(75,),layer_activation='tanh', **kwargs):
        activation_dict={'tanh':K.tanh,'relu':K.relu,'sigmoid':K.sigmoid}
        self.init = RandomNormal()
        self.hidden_layers=hidden_layers
        self.layer_activation=layer_activation
        self.activation=activation_dict[layer_activation]
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W=[]
        layers=[input_shape[-1]]+list(self.hidden_layers)+[1]
        for i in range(len(layers)-1):
            self.W.append(K.variable(self.init((layers[i],layers[i+1])),name='attention_W_'+str(i)))
        self.trainable_weights = self.W
        super(AttentionLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
	weights=self.feedforward(x)
	weights=K.squeeze(weights,axis=-1)
	weights=K.softmax(weights)
	return weights
	return K.batch_dot(x,weights,axes=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config={
        'hidden_layers':self.hidden_layers,
        'layer_activation':self.layer_activation
        }
        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def feedforward(self, x):
        activation=self.activation
        z=a=x
        for w in self.W:
	    z=K.dot(a,w)
            a=activation(z)
        return z

class AttentiveConcatenateLayer(Layer):
    def __init__(self, length=100, hidden_layers=(75,), layer_activation='tanh', **kwargs):
        activation_dict={'tanh':K.tanh,'relu':K.relu,'sigmoid':K.sigmoid}
        self.init = RandomNormal()
        self.hidden_layers=hidden_layers
        self.layer_activation=layer_activation
        self.activation=activation_dict[layer_activation]
        self.length=length
        super(AttentiveConcatenateLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W=[]
        layers=[2*input_shape[-1]]+list(self.hidden_layers)+[1]
        for i in range(len(layers)-1):
            self.W.append(K.variable(self.init((layers[i],layers[i+1])),name='attentive_concatenate_W_'+str(i)))
        self.trainable_weights = self.W
        super(AttentiveConcatenateLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
    	length=self.length
    	shape=K.int_shape(x)
    	a=K.repeat_elements(x,length,axis=1)
    	b=K.tile(x,[1,length,1])
    	x_concatenated=K.reshape(K.concatenate([a,b],axis=-1),(-1,length,length,shape[2]*2))
    	weights=self.feedforward(x_concatenated)
    	weights=K.reshape(weights,(-1,length))
    	weights=K.softmax(weights)
	return K.reshape(weights,(-1,length,length))
    	weighted_x_concatenated=x_concatenated*K.reshape(weights,(-1,length,length,1))
    	return K.sum(weighted_x_concatenated,axis=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] , 2*input_shape[-1])

    def get_config(self):
        config={
        'length':self.length,
        'hidden_layers':self.hidden_layers,
        'layer_activation':self.layer_activation
        }
        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def feedforward(self, x):
        activation=self.activation
        z=a=x
        for w in self.W:
            z=K.dot(a,w)
            a=activation(z)
        return z
