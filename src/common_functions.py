import numpy
from scipy import linalg
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from cis.deep.utils.theano import debug_print
from WPDefined import repeat_whole_matrix, repeat_whole_tensor
from word2embeddings.nn.util import zero_value, random_value_normal

def GRU_Combine_2Matrix(M1, M2, hidden_dim, U, W, b):
    #each row in matrix is a embeddings
    tensor1=M1.transpose().reshape((1, M1.shape[1], M1.shape[0])) #colmns wise embedding
    tensor2=M2.transpose().reshape((1, M2.shape[1], M2.shape[0]))
    raw_tensor=T.concatenate([tensor1, tensor2], axis=0)
    GRU_tensor_input=raw_tensor.dimshuffle((2,1,0))
    GRU_layer=GRU_Tensor3_Input_parallel(GRU_tensor_input, hidden_dim, U, W, b)
    GRUcombinedEMb=debug_print(GRU_layer.output_matrix.transpose(), 'GRUcombinedEMb') # hope each row is embedding
    return GRUcombinedEMb

def one_iteration(matrix, entity_Es, relation_Es, GRU_U, GRU_W, GRU_b, emb_size, entity_size, relation_size, entity_count, relation_count):   
    new_entity_E=T.zeros((entity_size, emb_size))  
    new_relation_E=T.zeros((relation_size, emb_size))  
    def forward_prop_step(triple, accu_entity_E, accu_relation_E):  
        accu_entity_E=debug_print(accu_entity_E, 'accu_entity_E_before_update')
        accu_relation_E=debug_print(accu_relation_E, 'accu_relation_E_before_update')
        triple=debug_print(triple, 'triple')        
        r_id=debug_print(triple[1], 'r_id')
        t_id=debug_print(triple[2], 't_id')
        head_E=entity_Es[triple[0]].reshape((emb_size,1))
        relation_E=relation_Es[r_id].reshape((emb_size,1))
        tail_E=entity_Es[t_id].reshape((emb_size,1))
        
        GUR_input=debug_print(T.concatenate([head_E, relation_E, tail_E], axis=1), 'GUR_input')
        GRU_layer=GRU_Triple_Input(X=GUR_input, word_dim=emb_size, hidden_dim=emb_size, U=GRU_U[r_id], W=GRU_W[r_id], b=GRU_b[r_id], bptt_truncate=-1)    
        new_r_emb=debug_print(GRU_layer.output_matrix[:,-2], 'new_r_emb')
        new_t_emb=debug_print(GRU_layer.output_vector_last, 'new_t_emb')
#         new_relation_E[r_id]+=GRU_layer.output_matrix[:,-2]
#         new_entity_E[t_id]+=GRU_layer.output_vector_last
        accu_relation_E=T.set_subtensor(accu_relation_E[r_id], accu_relation_E[r_id]+new_r_emb)
        accu_entity_E=T.set_subtensor(accu_entity_E[t_id], accu_entity_E[t_id]+new_t_emb)
        accu_entity_E=debug_print(accu_entity_E, 'accu_entity_E_after_update')
        accu_relation_E=debug_print(accu_relation_E, 'accu_relation_E_after_update')
        return accu_entity_E,accu_relation_E   # potential problem
    (entity_E_list, relation_E_list), updates = theano.scan(
        forward_prop_step,
        sequences=matrix,
        outputs_info=[new_entity_E,new_relation_E])
    
    entity_count=debug_print(entity_count.reshape((entity_size,1)), 'entity_count')
    relation_count=debug_print(relation_count.reshape((relation_size, 1)), 'relation_count')
    entity_E=debug_print(entity_E_list[-1]/entity_count+1e-6, 'new_entity_E') #to get rid of zero incoming info
    relation_E=debug_print(relation_E_list[-1]/relation_count, 'new_relation_E')
    return entity_E, relation_E

def ortho_weight(ndim):
    W=numpy.random.randn(ndim, ndim)
    u,s,v = numpy.linalg.svd(W)
    return u.astype('float64')

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout=nin
    if nout==nin and ortho:
        W=ortho_weight(nin)
    else:
        W = scale*numpy.random.randn(nin, nout)
    return W.astype('float64')

def create_nGRUs_para_Ramesh(rng, word_dim, hidden_dim, n):
        # Initialize the network parameters
        size=3*n*2
        list=[]
        for i in range(size):
            list.append(norm_weight(word_dim, hidden_dim).reshape((1, word_dim, hidden_dim)))
        weight_tensor=numpy.concatenate(list, axis=0)
#         U = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (n, 3, hidden_dim, word_dim))
#         W = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (n, 3, hidden_dim, hidden_dim))
        U=weight_tensor[:3*n]
        W=weight_tensor[3*n-1:]
        b = numpy.zeros((n, 3, hidden_dim))
        # Theano: Created shared variables
        U = theano.shared(name='U', value=U.astype(theano.config.floatX), borrow=True)
        W = theano.shared(name='W', value=W.astype(theano.config.floatX), borrow=True)
        b = theano.shared(name='b', value=b.astype(theano.config.floatX), borrow=True)
        return U, W, b
    
def create_nGRUs_para(rng, word_dim, hidden_dim, n):
        # Initialize the network parameters
        U = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (n, 3, hidden_dim, word_dim))
        W = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (n, 3, hidden_dim, hidden_dim))
        b = numpy.zeros((n, 3, hidden_dim))
        # Theano: Created shared variables
        U = theano.shared(name='U', value=U.astype(theano.config.floatX), borrow=True)
        W = theano.shared(name='W', value=W.astype(theano.config.floatX), borrow=True)
        b = theano.shared(name='b', value=b.astype(theano.config.floatX), borrow=True)
        return U, W, b


def create_GRU_para(rng, word_dim, hidden_dim):
        # Initialize the network parameters
        U = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, word_dim))
        W = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        b = numpy.zeros((3, hidden_dim))
        # Theano: Created shared variables
        U = theano.shared(name='U', value=U.astype(theano.config.floatX), borrow=True)
        W = theano.shared(name='W', value=W.astype(theano.config.floatX), borrow=True)
        b = theano.shared(name='b', value=b.astype(theano.config.floatX), borrow=True)
        return U, W, b
    
class GRU_Triple_Input(object):
    def __init__(self, X, word_dim, hidden_dim, U, W, b, bptt_truncate):
        X=X.transpose()
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        
        def forward_prop_step(x_t, s_t1_prev):            
            # GRU Layer 1
            z_t1 =debug_print( T.nnet.sigmoid(U[0].dot(x_t) + W[0].dot(s_t1_prev) + b[0]), 'z_t1')
            r_t1 = debug_print(T.nnet.sigmoid(U[1].dot(x_t) + W[1].dot(s_t1_prev) + b[1]), 'r_t1')
            c_t1 = debug_print(T.tanh(U[2].dot(x_t) + W[2].dot(s_t1_prev * r_t1) + b[2]), 'c_t1')
            s_t1 = debug_print((T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev, 's_t1')
            return s_t1
        
        s, updates = theano.scan(
            forward_prop_step,
            sequences=X[1:],
            truncate_gradient=self.bptt_truncate,
            outputs_info=dict(initial=X[0]))
        
        self.output_matrix=debug_print(s.transpose(), 'GRU_Matrix_Input.output_matrix')
        self.output_vector_mean=T.mean(self.output_matrix, axis=1)
        self.output_vector_max=T.max(self.output_matrix, axis=1)
        self.output_vector_last=self.output_matrix[:,-1]

class GRU_Matrix_Input(object):
    def __init__(self, X, word_dim, hidden_dim, U, W, b, bptt_truncate):
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        
        def forward_prop_step(x_t, s_t1_prev):            
            # GRU Layer 1
            z_t1 =debug_print( T.nnet.sigmoid(U[0].dot(x_t) + W[0].dot(s_t1_prev) + b[0]), 'z_t1')
            r_t1 = debug_print(T.nnet.sigmoid(U[1].dot(x_t) + W[1].dot(s_t1_prev) + b[1]), 'r_t1')
            c_t1 = debug_print(T.tanh(U[2].dot(x_t) + W[2].dot(s_t1_prev * r_t1) + b[2]), 'c_t1')
            s_t1 = debug_print((T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev, 's_t1')
            return s_t1
        
        s, updates = theano.scan(
            forward_prop_step,
            sequences=X.transpose(1,0),
            truncate_gradient=self.bptt_truncate,
            outputs_info=dict(initial=T.zeros(self.hidden_dim)))
        
        self.output_matrix=debug_print(s.transpose(), 'GRU_Matrix_Input.output_matrix')
        self.output_vector_mean=T.mean(self.output_matrix, axis=1)
        self.output_vector_max=T.max(self.output_matrix, axis=1)
        self.output_vector_last=self.output_matrix[:,-1]

class GRU_Tensor3_Input_parallel(object):
    def __init__(self, Tensor, hidden_dim, U, W, b):
        #hope to address it in parallel
        self.hidden_dim = hidden_dim
        
        def forward_prop_step(x_t, s_t1_prev):            
            # GRU Layer 1
            z_t1 =debug_print( T.nnet.sigmoid(U[0].dot(x_t) + W[0].dot(s_t1_prev) + b[0].reshape((b.shape[1],1))), 'z_t1')
            r_t1 = debug_print(T.nnet.sigmoid(U[1].dot(x_t) + W[1].dot(s_t1_prev) + b[1].reshape((b.shape[1],1))), 'r_t1')
            c_t1 = debug_print(T.tanh(U[2].dot(x_t) + W[2].dot(s_t1_prev * r_t1) + b[2].reshape((b.shape[1],1))), 'c_t1')
            s_t1 = debug_print((T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev, 's_t1')
            return s_t1
        
        new_T, updates = theano.scan(
            forward_prop_step,
            sequences=Tensor.dimshuffle(2,1,0),
            outputs_info=dict(initial=T.zeros((self.hidden_dim, Tensor.shape[0]))))
        
#         self.output_matrix=debug_print(s.transpose(), 'GRU_Matrix_Input.output_matrix')
        self.output_tensor=new_T.dimshuffle(2,1,0)
        self.output_matrix=new_T[-1]#column wise embedding
class GRU_Tensor3_Triple_Input(object):
    def __init__(self, T, hidden_dim, U, W, b):
        T=debug_print(T,'T')
        def recurrence(matrix):
            sub_matrix=debug_print(matrix, 'sub_matrix')
            GRU_layer=GRU_Triple_Input(sub_matrix, sub_matrix.shape[0], hidden_dim,U,W,b, -1)
            return GRU_layer.output_vector_mean
        new_M, updates = theano.scan(recurrence,
                                     sequences=T,
                                     outputs_info=None)
        self.output=debug_print(new_M.transpose(), 'GRU_Tensor3_Input.output')

class GRU_Tensor3_Input(object):
    def __init__(self, T, hidden_dim, U, W, b):
        T=debug_print(T,'T')
        def recurrence(matrix):
            sub_matrix=debug_print(matrix, 'sub_matrix')
            GRU_layer=GRU_Matrix_Input(sub_matrix, sub_matrix.shape[0], hidden_dim,U,W,b, -1)
            return GRU_layer.output_vector_mean
        new_M, updates = theano.scan(recurrence,
                                     sequences=T,
                                     outputs_info=None)
        self.output=debug_print(new_M.transpose(), 'GRU_Tensor3_Input.output')
        
def create_params_WbWAE(input_dim, output_dim):
    W = numpy.random.uniform(-numpy.sqrt(1./output_dim), numpy.sqrt(1./output_dim), (6, output_dim, input_dim))
    w = numpy.random.uniform(-numpy.sqrt(1./output_dim), numpy.sqrt(1./output_dim), (1,output_dim))

    W = theano.shared(name='W', value=W.astype(theano.config.floatX))
    w = theano.shared(name='w', value=w.astype(theano.config.floatX))
    
    return W, w

class Word_by_Word_Attention_EntailmentPaper(object):
    def __init__(self, l_hidden_M, r_hidden_M, W_y,W_h,W_r, w, W_t, W_p, W_x, r_dim):
        self.Y=l_hidden_M
        self.H=r_hidden_M
        self.attention_dim=r_dim
        self.r0 = theano.shared(name='r0', value=numpy.zeros(self.attention_dim, dtype=theano.config.floatX))
        def loop(h_t, r_t_1):
            M_t=T.tanh(W_y.dot(self.Y)+(W_h.dot(h_t)+W_r.dot(r_t_1)).dimshuffle(0,'x'))
            alpha_t=T.nnet.softmax(w.dot(M_t))
            r_t=self.Y.dot(alpha_t.reshape((self.Y.shape[1],1)))+T.tanh(W_t.dot(r_t_1))

            r_t=T.sum(M_t, axis=1)
            return r_t
        
        r, updates= theano.scan(loop,
                                sequences=self.H.transpose(),
                                outputs_info=self.r0
                                )
        
        H_star=T.tanh(W_p.dot(r[-1]+W_x.dot(self.H[:,-1])))
        self.output=H_star    
def create_conv_para(rng, filter_shape):
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        return W, b

class Conv_with_input_para(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, W, b):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = W
        self.b = b

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='valid')    #here, we should pad enough zero padding for input 
        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        conv_with_bias = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        narrow_conv_out=conv_with_bias.reshape((image_shape[0], 1, filter_shape[0], image_shape[3]-filter_shape[3]+1)) #(batch, 1, kernerl, ishape[1]-filter_size1[1]+1)
        
        #pad filter_size-1 zero embeddings at both sides
        left_padding = T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        right_padding = T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        self.output = T.concatenate([left_padding, narrow_conv_out, right_padding], axis=3) 
        

        # store parameters of this layer
        self.params = [self.W, self.b]

class Conv(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='valid')    #here, we should pad enough zero padding for input 
        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        conv_with_bias = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        narrow_conv_out=conv_with_bias.reshape((image_shape[0], 1, filter_shape[0], image_shape[3]-filter_shape[3]+1)) #(batch, 1, kernerl, ishape[1]-filter_size1[1]+1)
        
        #pad filter_size-1 zero embeddings at both sides
        left_padding = 1e-20+T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        right_padding = 1e-20+T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        self.output = T.concatenate([left_padding, narrow_conv_out, right_padding], axis=3) 
        

        # store parameters of this layer
        self.params = [self.W, self.b]

class Average_Pooling(object):
    """this pooling includes all-ap and w-ap, it can output two vectors as well as two matrices"""

    def __init__(self, rng, input_l, input_r, kern, left_l, right_l, left_r, right_r, length_l, length_r, dim, window_size, maxSentLength): # length_l, length_r: valid lengths after conv


        fan_in = kern #kern numbers
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = kern
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(kern, kern)),
            dtype=theano.config.floatX),
                               borrow=True) #a weight matrix kern*kern
        
        input_l_matrix=input_l.reshape((input_l.shape[2], input_l.shape[3]))
        input_l_matrix=input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)]
        input_r_matrix=input_r.reshape((input_r.shape[2], input_r.shape[3]))
        input_r_matrix=input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)]
        
        
        simi_tensor=compute_simi_feature_batch1_new(input_l_matrix, input_r_matrix, length_l, length_r, self.W, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(T.sum(simi_tensor, axis=1).reshape((1, length_l)),'simi_question')
        simi_answer=debug_print(T.sum(simi_tensor, axis=0).reshape((1, length_r)), 'simi_answer')
        
#         weights_question =T.nnet.softmax(simi_question) 
#         weights_answer=T.nnet.softmax(simi_answer) 
#         weights_question =simi_question
#         weights_answer=simi_answer        
        weights_question =simi_question/T.sum(simi_question)
        weights_answer=simi_answer/T.sum(simi_answer) 
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))
        
        weights_question_matrix=T.repeat(weights_question, kern, axis=0)
        weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)
        
        #with attention
        weighted_matrix_l=input_l_matrix*weights_question_matrix # first add 1e-20 for each element to make non-zero input for weight gradient
        weighted_matrix_r=input_r_matrix*weights_answer_matrix
        '''
        #without attention
        weighted_matrix_l=input_l_matrix # first add 1e-20 for each element to make non-zero input for weight gradient
        weighted_matrix_r=input_r_matrix
        '''
        
        sub_tensor_list_l=[]
        for i in range(window_size):
            if i ==0:
                sub_tensor_list_l.append(weighted_matrix_l.dimshuffle(('x', 0, 1)))
            else:
                sub_tensor_list_l.append(T.concatenate([weighted_matrix_l[:, i:], weighted_matrix_l[:, :i]], axis=1).dimshuffle(('x', 0, 1)))
        sub_tensor_l=T.concatenate(sub_tensor_list_l, axis=0)
        average_pooled_matrix_l=T.sum(sub_tensor_l, axis=0)[:, :1-window_size]
        average_pooled_tensor_l=average_pooled_matrix_l.reshape((input_l.shape[0], input_l.shape[1], input_l.shape[2], length_l-window_size+1))
        
        #pad filter_size-1 zero embeddings at both sides
        left_padding = T.zeros((input_l.shape[0], input_l.shape[1], input_l.shape[2], left_l), dtype=theano.config.floatX)
        right_padding = T.zeros((input_l.shape[0], input_l.shape[1], input_l.shape[2], right_l), dtype=theano.config.floatX)
        
        self.output_tensor_l = T.concatenate([left_padding, average_pooled_tensor_l, right_padding], axis=3) 
        

        sub_tensor_list_r=[]
        for i in range(window_size):
            if i ==0:
                sub_tensor_list_r.append(weighted_matrix_r.dimshuffle(('x', 0, 1)))
            else:
                sub_tensor_list_r.append(T.concatenate([weighted_matrix_r[:, i:], weighted_matrix_r[:, :i]], axis=1).dimshuffle(('x', 0, 1)))
        sub_tensor_r=T.concatenate(sub_tensor_list_r, axis=0)
        average_pooled_matrix_r=T.sum(sub_tensor_r, axis=0)[:, :1-window_size]        
        average_pooled_tensor_r=average_pooled_matrix_r.reshape((input_r.shape[0], input_r.shape[1], input_r.shape[2], length_r-window_size+1))
        #pad filter_size-1 zero embeddings at both sides
        left_padding = T.zeros((input_r.shape[0], input_r.shape[1], input_r.shape[2], left_r), dtype=theano.config.floatX)
        right_padding = T.zeros((input_r.shape[0], input_r.shape[1], input_r.shape[2], right_r), dtype=theano.config.floatX)
        
        self.output_tensor_r = T.concatenate([left_padding, average_pooled_tensor_r, right_padding], axis=3) 


        dot_l=T.sum(weighted_matrix_l, axis=1) # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=T.sum(weighted_matrix_r, axis=1)        
        norm_l=T.sqrt((dot_l**2).sum())
        norm_r=T.sqrt((dot_r**2).sum())
        
        self.output_vector_l=(dot_l/norm_l).reshape((1, kern))
        self.output_vector_r=(dot_r/norm_r).reshape((1, kern))      
        self.output_concate=T.concatenate([dot_l, dot_r], axis=0).reshape((1, kern*2))
        self.output_cosine=(T.sum(dot_l*dot_r)/norm_l/norm_r).reshape((1,1))
        
        '''
        dot_l=T.sum(input_l_matrix, axis=1) # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=T.sum(input_r_matrix, axis=1)        
        '''
        self.output_eucli=debug_print(T.sqrt(T.sqr(dot_l-dot_r).sum()+1e-20).reshape((1,1)),'output_eucli')
        self.output_eucli_to_simi=1.0/(1.0+self.output_eucli)
        #self.output_simi=self.output_eucli
        
        self.output_attention_vector_l=weights_question
        self.output_attention_vector_r=weights_answer
        self.output_attention_matrix=simi_tensor        

        self.params = [self.W]

class Average_Pooling_for_ARCII(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_l, input_r): # length_l, length_r: valid lengths after conv
        
        input_l_matrix=debug_print(input_l.reshape((input_l.shape[2], input_l.shape[3])), 'origin_input_l_matrix')
        #input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')
        input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')
        #input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')
    
        
        #with attention
        dot_l=debug_print(T.sum(input_l_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=debug_print(T.sum(input_r_matrix, axis=1),'dot_r')      
        '''
        #without attention
        dot_l=debug_print(T.sum(input_l_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=debug_print(T.sum(input_r_matrix, axis=1),'dot_r')      
        '''
        '''
        #with attention, then max pooling
        dot_l=debug_print(T.max(input_l_matrix*weights_question_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=debug_print(T.max(input_r_matrix*weights_answer_matrix, axis=1),'dot_r')          
        '''
        norm_l=debug_print(T.sqrt((dot_l**2).sum()),'norm_l')
        norm_r=debug_print(T.sqrt((dot_r**2).sum()), 'norm_r')
        
        self.output_vector_l=debug_print((dot_l/norm_l).reshape((1, input_l.shape[2])),'output_vector_l')
        self.output_vector_r=debug_print((dot_r/norm_r).reshape((1, input_r.shape[2])), 'output_vector_r')      
        self.output_concate=T.concatenate([dot_l, dot_r], axis=0).reshape((1, input_l.shape[2]*2))
        self.output_cosine=debug_print((T.sum(dot_l*dot_r)/norm_l/norm_r).reshape((1,1)),'output_cosine')

        self.output_eucli=debug_print(T.sqrt(T.sqr(dot_l-dot_r).sum()+1e-20).reshape((1,1)),'output_eucli')
        self.output_eucli_to_simi=1.0/(1.0+self.output_eucli)
        #self.output_eucli_to_simi_exp=1.0/T.exp(self.output_eucli) # not good
        #self.output_sigmoid_simi=debug_print(T.nnet.sigmoid(T.dot(dot_l/norm_l, (dot_r/norm_r).T)).reshape((1,1)),'output_sigmoid_simi')    


class Average_Pooling_for_Top(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_l, input_r, kern, left_l, right_l, left_r, right_r, length_l, length_r, dim): # length_l, length_r: valid lengths after conv



        fan_in = kern #kern numbers
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = kern
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(kern, kern)),
            dtype=theano.config.floatX),
                               borrow=True) #a weight matrix kern*kern
        
        input_l_matrix=debug_print(input_l.reshape((input_l.shape[2], input_l.shape[3])), 'origin_input_l_matrix')
        input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')
        input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')
        input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')
        
        
        simi_tensor=compute_simi_feature_batch1_new(input_l_matrix, input_r_matrix, length_l, length_r, self.W, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(T.sum(simi_tensor, axis=1).reshape((1, length_l)),'simi_question')
        simi_answer=debug_print(T.sum(simi_tensor, axis=0).reshape((1, length_r)), 'simi_answer')
        
#         weights_question =T.nnet.softmax(simi_question) 
#         weights_answer=T.nnet.softmax(simi_answer) 
        #weights_question =simi_question
        #weights_answer=simi_answer        
        weights_question =simi_question/T.sum(simi_question)
        weights_answer=simi_answer/T.sum(simi_answer)    
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))
        
        weights_question_matrix=T.repeat(weights_question, kern, axis=0)
        weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)
        
        #with attention
        dot_l=debug_print(T.sum(input_l_matrix*weights_question_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=debug_print(T.sum(input_r_matrix*weights_answer_matrix, axis=1),'dot_r')      
        '''
        #without attention
        dot_l=debug_print(T.sum(input_l_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=debug_print(T.sum(input_r_matrix, axis=1),'dot_r')      
        '''
        '''
        #with attention, then max pooling
        dot_l=debug_print(T.max(input_l_matrix*weights_question_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=debug_print(T.max(input_r_matrix*weights_answer_matrix, axis=1),'dot_r')          
        '''
        norm_l=debug_print(T.sqrt((dot_l**2).sum()),'norm_l')
        norm_r=debug_print(T.sqrt((dot_r**2).sum()), 'norm_r')
        
        self.output_vector_l=debug_print((dot_l/norm_l).reshape((1, kern)),'output_vector_l')
        self.output_vector_r=debug_print((dot_r/norm_r).reshape((1, kern)), 'output_vector_r')      
        self.output_concate=T.concatenate([dot_l, dot_r], axis=0).reshape((1, kern*2))
        self.output_cosine=debug_print((T.sum(dot_l*dot_r)/norm_l/norm_r).reshape((1,1)),'output_cosine')

        self.output_eucli=debug_print(T.sqrt(T.sqr(dot_l-dot_r).sum()+1e-20).reshape((1,1)),'output_eucli')
        self.output_eucli_to_simi=1.0/(1.0+self.output_eucli)
        #self.output_eucli_to_simi_exp=1.0/T.exp(self.output_eucli) # not good
        #self.output_sigmoid_simi=debug_print(T.nnet.sigmoid(T.dot(dot_l/norm_l, (dot_r/norm_r).T)).reshape((1,1)),'output_sigmoid_simi')    
        self.output_attentions=unify_eachone(simi_tensor, length_l, length_r, 4)
        
        self.output_attention_vector_l=weights_question
        self.output_attention_vector_r=weights_answer
        self.output_attention_matrix=simi_tensor
        

        self.params = [self.W]

def compute_simi_feature_batch1_new(input_l_matrix, input_r_matrix, length_l, length_r, para_matrix, dim):
    #matrix_r_after_translate=debug_print(T.dot(para_matrix, input_r_matrix), 'matrix_r_after_translate')
    matrix_r_after_translate=input_r_matrix
    
    input_l_tensor=input_l_matrix.dimshuffle('x',0,1)
    input_l_tensor=T.repeat(input_l_tensor, dim, axis=0)[:length_r,:,:]
    input_l_tensor=input_l_tensor.dimshuffle(2,1,0).dimshuffle(0,2,1)
    repeated_1=input_l_tensor.reshape((length_l*length_r, input_l_matrix.shape[0])).dimshuffle(1,0)
    
    input_r_tensor=matrix_r_after_translate.dimshuffle('x',0,1)
    input_r_tensor=T.repeat(input_r_tensor, dim, axis=0)[:length_l,:,:]
    input_r_tensor=input_r_tensor.dimshuffle(0,2,1)
    repeated_2=input_r_tensor.reshape((length_l*length_r, matrix_r_after_translate.shape[0])).dimshuffle(1,0)
    
    #wrong
    #repeated_1=debug_print(T.repeat(input_l_matrix, dim, axis=1)[:, : (length_l*length_r)],'repeated_1') # add 10 because max_sent_length is only input for conv, conv will make size bigger
    #repeated_2=debug_print(repeat_whole_tensor(matrix_r_after_translate, dim, False)[:, : (length_l*length_r)],'repeated_2')
    '''
    #cosine attention   
    length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=0)),'length_1')
    length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')

    multi=debug_print(repeated_1*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')
    
    list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
    simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')
    
    '''
    #euclid, effective for wikiQA
    gap=debug_print(repeated_1-repeated_2, 'gap')
    eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
    simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((length_l, length_r)), 'simi_matrix')
    
    
    return simi_matrix#[:length_l, :length_r]

def compute_simi_feature_batch1(input_l_matrix, input_r_matrix, length_l, length_r, para_matrix, dim):
    #matrix_r_after_translate=debug_print(T.dot(para_matrix, input_r_matrix), 'matrix_r_after_translate')
    matrix_r_after_translate=input_r_matrix

    repeated_1=debug_print(T.repeat(input_l_matrix, dim, axis=1)[:, : (length_l*length_r)],'repeated_1') # add 10 because max_sent_length is only input for conv, conv will make size bigger
    repeated_2=debug_print(repeat_whole_tensor(matrix_r_after_translate, dim, False)[:, : (length_l*length_r)],'repeated_2')
    '''
    #cosine attention   
    length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=0)),'length_1')
    length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')

    multi=debug_print(repeated_1*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')
    
    list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
    simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')
    
    '''
    #euclid, effective for wikiQA
    gap=debug_print(repeated_1-repeated_2, 'gap')
    eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
    simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((length_l, length_r)), 'simi_matrix')
    
    
    return simi_matrix#[:length_l, :length_r]

def compute_simi_feature(tensor, dim, para_matrix):
    odd_tensor=debug_print(tensor[0:tensor.shape[0]:2,:,:,:],'odd_tensor')
    even_tensor=debug_print(tensor[1:tensor.shape[0]:2,:,:,:], 'even_tensor')
    even_tensor_after_translate=debug_print(T.dot(para_matrix, 1e-20+even_tensor.reshape((tensor.shape[2], dim*tensor.shape[0]/2))), 'even_tensor_after_translate')
    fake_even_tensor=debug_print(even_tensor_after_translate.reshape((tensor.shape[0]/2, tensor.shape[1], tensor.shape[2], tensor.shape[3])),'fake_even_tensor')

    repeated_1=debug_print(T.repeat(odd_tensor, dim, axis=3),'repeated_1')
    repeated_2=debug_print(repeat_whole_matrix(fake_even_tensor, dim, False),'repeated_2')
    #repeated_2=T.repeat(even_tensor, even_tensor.shape[3], axis=2).reshape((tensor.shape[0]/2, tensor.shape[1], tensor.shape[2], tensor.shape[3]**2))    
    length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=2)),'length_1')
    length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=2)), 'length_2')

    multi=debug_print(repeated_1*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=2),'sum_multi')
    
    list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
    
    return list_of_simi.reshape((tensor.shape[0]/2, tensor.shape[1], tensor.shape[3], tensor.shape[3]))

def compute_acc(label_list, scores_list):
    #label_list contains 0/1, 500 as a minibatch, score_list contains score between -1 and 1, 500 as a minibatch
    if len(label_list)%500!=0 or len(scores_list)%500!=0:
        print 'len(label_list)%500: ', len(label_list)%500, ' len(scores_list)%500: ', len(scores_list)%500
        exit(0)
    if len(label_list)!=len(scores_list):
        print 'len(label_list)!=len(scores_list)', len(label_list), ' and ',len(scores_list)
        exit(0)
    correct_count=0
    total_examples=len(label_list)/500
    start_posi=range(total_examples)*500
    for i in start_posi:
        set_1=set()
        
        for scan in range(i, i+500):
            if label_list[scan]==1:
                set_1.add(scan)
        set_0=set(range(i, i+500))-set_1
        flag=True
        for zero_posi in set_0:
            for scan in set_1:
                if scores_list[zero_posi]> scores_list[scan]:
                    flag=False
        if flag==True:
            correct_count+=1
    
    return correct_count*1.0/total_examples
#def unify_eachone(tensor, left1, right1, left2, right2, dim, Np):
def top_k_pooling(matrix, sentlength_1, sentlength_2, Np):

    #tensor: (1, feature maps, 66, 66)
    #sentlength_1=dim-left1-right1
    #sentlength_2=dim-left2-right2
    #core=tensor[:,:, left1:(dim-right1),left2:(dim-right2) ]
    '''
    repeat_row=Np/sentlength_1
    extra_row=Np%sentlength_1
    repeat_col=Np/sentlength_2
    extra_col=Np%sentlength_2    
    '''
    #repeat core
    matrix_1=repeat_whole_tensor(matrix, 5, True) 
    matrix_2=repeat_whole_tensor(matrix_1, 5, False)

    list_values=matrix_2.flatten()
    neighborsArgSorted = T.argsort(list_values)
    kNeighborsArg = neighborsArgSorted[-(Np**2):]    
    top_k_values=list_values[kNeighborsArg]
    

    all_max_value=top_k_values.reshape((1, Np**2))
    
    return all_max_value  
def unify_eachone(matrix, sentlength_1, sentlength_2, Np):

    #tensor: (1, feature maps, 66, 66)
    #sentlength_1=dim-left1-right1
    #sentlength_2=dim-left2-right2
    #core=tensor[:,:, left1:(dim-right1),left2:(dim-right2) ]

    repeat_row=Np/sentlength_1
    extra_row=Np%sentlength_1
    repeat_col=Np/sentlength_2
    extra_col=Np%sentlength_2    

    #repeat core
    matrix_1=repeat_whole_tensor(matrix, 5, True) 
    matrix_2=repeat_whole_tensor(matrix_1, 5, False)
    
    new_rows=T.maximum(sentlength_1, sentlength_1*repeat_row+extra_row)
    new_cols=T.maximum(sentlength_2, sentlength_2*repeat_col+extra_col)
    
    #core=debug_print(core_2[:,:, :new_rows, : new_cols],'core')
    new_matrix=debug_print(matrix_2[:new_rows,:new_cols], 'new_matrix')
    #determine x, y start positions
    size_row=new_rows/Np
    remain_row=new_rows%Np
    size_col=new_cols/Np
    remain_col=new_cols%Np
    
    xx=debug_print(T.concatenate([T.arange(Np-remain_row+1)*size_row, (Np-remain_row)*size_row+(T.arange(remain_row)+1)*(size_row+1)]),'xx')
    yy=debug_print(T.concatenate([T.arange(Np-remain_col+1)*size_col, (Np-remain_col)*size_col+(T.arange(remain_col)+1)*(size_col+1)]),'yy')
    
    list_of_maxs=[]
    for i in xrange(Np):
        for j in xrange(Np):
            region=debug_print(new_matrix[xx[i]:xx[i+1], yy[j]:yy[j+1]],'region')
            #maxvalue1=debug_print(T.max(region, axis=2), 'maxvalue1')
            maxvalue=debug_print(T.max(region).reshape((1,1)), 'maxvalue')
            list_of_maxs.append(maxvalue)
    

    all_max_value=T.concatenate(list_of_maxs, axis=1).reshape((1, Np**2))
    
    return all_max_value            


class Create_Attention_Input_Cnn(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, tensor_l, tensor_r, dim,kern, l_left_pad, l_right_pad, r_left_pad, r_right_pad): # length_l, length_r: valid lengths after conv
        #first reshape into matrix
        matrix_l=tensor_l.reshape((tensor_l.shape[2], tensor_l.shape[3]))
        matrix_r=tensor_r.reshape((tensor_r.shape[2], tensor_r.shape[3]))
        #start
        repeated_1=debug_print(T.repeat(matrix_l, dim, axis=1),'repeated_1') # add 10 because max_sent_length is only input for conv, conv will make size bigger
        repeated_2=debug_print(repeat_whole_tensor(matrix_r, dim, False),'repeated_2')
        '''
        #cosine attention   
        length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=0)),'length_1')
        length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')
    
        multi=debug_print(repeated_1*repeated_2, 'multi')
        sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')
        
        list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
        simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')
        
        '''
        #euclid, effective for wikiQA
        gap=debug_print(repeated_1-repeated_2, 'gap')
        eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
        simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((dim, dim)), 'simi_matrix')
        W_bound = numpy.sqrt(6. / (dim + kern))
        self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(kern, dim)),dtype=theano.config.floatX),borrow=True) #a weight matrix kern*kern
        matrix_l_attention=debug_print(T.dot(self.W, simi_matrix.T), 'matrix_l_attention')
        matrix_r_attention=debug_print(T.dot(self.W, simi_matrix), 'matrix_r_attention')
        #reset zero at both side
        left_zeros_l=T.set_subtensor(matrix_l_attention[:,:l_left_pad], T.zeros((matrix_l_attention.shape[0], l_left_pad), dtype=theano.config.floatX))
        right_zeros_l=T.set_subtensor(left_zeros_l[:,-l_right_pad:], T.zeros((matrix_l_attention.shape[0], l_right_pad), dtype=theano.config.floatX))
        left_zeros_r=T.set_subtensor(matrix_r_attention[:,:r_left_pad], T.zeros((matrix_r_attention.shape[0], r_left_pad), dtype=theano.config.floatX))
        right_zeros_r=T.set_subtensor(left_zeros_r[:,-r_right_pad:], T.zeros((matrix_r_attention.shape[0], r_right_pad), dtype=theano.config.floatX))       
        #combine with original input matrix
        self.new_tensor_l=T.concatenate([matrix_l,right_zeros_l], axis=0).reshape((tensor_l.shape[0], 2*tensor_l.shape[1], tensor_l.shape[2], tensor_l.shape[3])) 
        self.new_tensor_r=T.concatenate([matrix_r,right_zeros_r], axis=0).reshape((tensor_r.shape[0], 2*tensor_r.shape[1], tensor_r.shape[2], tensor_r.shape[3])) 
        
        self.params=[self.W]

def Diversify_Reg(W):
    loss=((W.dot(W.T)-T.eye(n=W.shape[0], m=W.shape[0], k=0, dtype=theano.config.floatX))**2).sum()  
    return loss      
    
def Determinant(W):
    prod=W.dot(W.T)
    loss=1.0/T.log(theano.tensor.nlinalg.Det()(prod))
    return loss    