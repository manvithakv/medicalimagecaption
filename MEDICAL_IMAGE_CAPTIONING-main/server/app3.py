import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io,json
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications import DenseNet121
from keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Softmax, RNN, Dense, Embedding, LSTM,Layer,Dropout,GRU
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@tf.keras.utils.register_keras_serializable()

class Encoder(tf.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns output sequence
    '''

    def __init__(self,units):
        super().__init__()
        self.units = units
        self.dense = Dense(self.units,name = 'Enc_dense')


    def call(self,img):
      '''
          This function takes a sequence input and the initial states of the encoder.
          Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to encoder_lstm
          returns -- All encoder_outputs, last time steps hidden and cell state
      '''
      #enc_out = self.maxpool(tf.expand_dims(img,axis = 2))
      enc_out = self.dense(img)
      return enc_out


    def initialize_states(self,batch_size):
      '''
      Given a batch size it will return intial hidden state and intial cell state.
      If batch size is 32- Hidden state is zeros of size [32,lstm_units], cell state zeros is of size [32,lstm_units]
      '''
      self.batch_size  = batch_size

      self.enc_h =tf.zeros((self.batch_size, self.units))

      #self.enc_c = tf.zeros((self.batch_size, self.lstm_size))
      return self.enc_h

@tf.keras.utils.register_keras_serializable()

class Attention(tf.keras.layers.Layer):
  '''
    Class the calculates score based on the scoring_function using Bahdanu attention mechanism.
  '''
  def __init__(self,att_units):
    # Please go through the reference notebook and research paper to complete the scoring functions
    super().__init__()

    self.att_units = att_units

    self.w1 =  tf.keras.layers.Dense( self.att_units , name = 'w1')
    self.w2 =  tf.keras.layers.Dense( self.att_units,name = 'w2')
    self.v =  tf.keras.layers.Dense(1,name = 'v')

  def call(self,decoder_hidden_state,encoder_output):
    '''
      Attention mechanism takes two inputs current step -- decoder_hidden_state and all the encoder_outputs.
      * Based on the scoring function we will find the score or similarity between decoder_hidden_state and encoder_output.
        Multiply the score function with your encoder_outputs to get the context vector.
        Function returns context vector and attention weights(softmax - scores)
    '''
    self.decoder_hidden_state = decoder_hidden_state
    self.encoder_output = encoder_output


    self.decoder_hidden_state = tf.expand_dims(self.decoder_hidden_state,axis = 1)
    score = self.v(tf.nn.tanh(
              self.w1(self.decoder_hidden_state) + self.w2(self.encoder_output)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * self.encoder_output
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector,attention_weights


@tf.keras.utils.register_keras_serializable()

class OneStepDecoder(tf.keras.Model):
  def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units  ,att_units):

      # Initialize decoder embedding layer, LSTM and any other objects needed
      super().__init__()
      self.tar_vocab_size = tar_vocab_size
      self.embedding_dim = embedding_dim
      self.input_length = input_length
      self.dec_units = dec_units
      self.att_units = att_units
      self.dec_emb = Embedding(tar_vocab_size,embedding_dim,trainable = True , name = 'dec_embb')
      self.dec_lstm = GRU(self.dec_units, return_state=True, return_sequences=True, name="Decoder_LSTM")
      self.dense   = Dense(self.tar_vocab_size, name = 'one_dec')
      self.attention=Attention( self.att_units)
      self.d1 = Dropout(0.3,name = 'd1')
      self.d2 = Dropout(0.3,name = 'd2')
      self.d3 = Dropout(0.3,name = 'd3')

  @tf.function
  def call(self,input_to_decoder, encoder_output, state_h):
    '''
        One step decoder mechanisim step by step:
      A. Pass the input_to_decoder to the embedding layer and then get the output(batch_size,1,embedding_dim)
      B. Using the encoder_output and decoder hidden state, compute the context vector.
      C. Concat the context vector with the step A output
      D. Pass the Step-C output to LSTM/GRU and get the decoder output and states(hidden and cell state)
      E. Pass the decoder output to dense layer(vocab size) and store the result into output.
      F. Return the states from step D, output from Step E, attention weights from Step -B
    '''
    self.input_to_decoder = input_to_decoder
    self.encoder_output = encoder_output
    self.state_h = state_h

    #A
    target_embedd           = self.dec_emb (self.input_to_decoder)     #(batch_size,1,embedingdim)
    #B
    target_embedd = self.d1(target_embedd)

    context_vector,attention_weights=self.attention(self.state_h,self.encoder_output) #context vector shape = (batch_size,att_units)
    #C
    concated = tf.concat([  tf.expand_dims(context_vector, 1),target_embedd], -1)
    concated = self.d2(concated)

    #D
    lstm_output, hs      = self.dec_lstm(concated, initial_state=self.state_h)

    lstm_output = tf.reshape(lstm_output, (-1, lstm_output.shape[2]))
    lstm_output = self.d3(lstm_output)
    #E
    op = self.dense(lstm_output)
    #op = tf.squeeze(op,[1])
    return op,hs,attention_weights,context_vector

@tf.keras.utils.register_keras_serializable()

class Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, input_length, dec_units ,att_units):
      #Intialize necessary variables and create an object from the class onestepdecoder
      super().__init__()
      self.out_vocab_size = out_vocab_size
      self.embedding_dim = embedding_dim
      self.input_length = input_length
      self.dec_units = dec_units
      self.att_units = att_units
      self.onestep = OneStepDecoder(self.out_vocab_size,self.embedding_dim ,self.input_length,self.dec_units,self.att_units)

    @tf.function
    def call(self, input_to_decoder,encoder_output,decoder_hidden_state):


        #Initialize an empty Tensor array, that will store the outputs at each and every time step
        #Create a tensor array as shown in the reference notebook

        #Iterate till the length of the decoder input
            # Call onestepdecoder for each token in decoder_input
            # Store the output in tensorarray
        # Return the tensor array

        all_outputs = tf.TensorArray(tf.float32,size =input_to_decoder.shape[1],name = 'output_arrays' )
        self.input_to_decoder = input_to_decoder
        self.encoder_output = encoder_output
        self.decoder_hidden_state = decoder_hidden_state

        for timestep in tf.range(input_to_decoder.shape[1]):
          op,hs,attention_weights,context_vector = self.onestep(self.input_to_decoder[:,timestep:timestep+1], self.encoder_output, self.decoder_hidden_state)
          self.decoder_hidden_state = hs
          all_outputs = all_outputs.write(timestep,op)
        all_outputs = tf.transpose(all_outputs.stack(),[1,0,2])
        return all_outputs


@tf.keras.utils.register_keras_serializable()
class encoder_decoder(tf.keras.Model):
  #def _init_(self,#params):
    #Intialize objects from encoder decoder
  def __init__(self,out_vocab_size , embedding_size_d, input_length_d,lstm_size_d,att_units,batch_size,units):

        #Create encoder object
        #Create decoder object
        #Intialize Dense layer(out_vocab_size) with activation='softmax'

        super().__init__()

        self.units = units
        self.out_vocab_size = out_vocab_size
        self.embedding_size_d = embedding_size_d
        self.lstm_size_d = lstm_size_d
        self.input_length_d = input_length_d
        self.batch_size = batch_size
        self.att_units = att_units

        self.encoder = Encoder(self.units)

        self.decoder = Decoder(out_vocab_size , embedding_size_d, input_length_d,lstm_size_d,att_units )
        #self.dense   = TimeDistributed(Dense(self.out_vocab_size, activation='softmax'))
        self.dense   = Dense(self.out_vocab_size,name = 'enc_dec_dense')



  def call(self,data):
    #Intialize encoder states, Pass the encoder_sequence to the embedding layer
    # Decoder initial states are encoder final states, Initialize it accordingly
    # Pass the decoder sequence,encoder_output,decoder states to Decoder
    # return the decoder output
    self.inputs,self.outputs = data[0], data[1]
    print("="*20, "ENCODER", "="*20)
    self.encoder_h= self.encoder.initialize_states(self.batch_size)
    self.encoder_output = self.encoder(self.inputs)
    print("-"*27)
    print("ENCODER ==> OUTPUT SHAPE",self.encoder_output.shape)
    print("ENCODER ==> HIDDEN STATE SHAPE",self.encoder_h.shape)
    print("="*20, "DECODER", "="*20)
    output= self.decoder(self.outputs,self.encoder_output,self.encoder_h)
    print("-"*27)
    print("FINAL OUTPUT SHAPE",output.shape)
    print("="*50)
    return output




from keras.preprocessing import image as keras_image

with open('imp1.json', 'r') as f:
    imp1 = json.load(f)

with open('imp2.json', 'r') as f:
    imp2 = json.load(f)

  
chex_weights = 'C:/Users/Public/mernproject2/server/chexweights.h5'
chexnet = DenseNet121(weights=chex_weights,                    
                      classes = 14,input_shape=(224,224,3))
model = Model(chexnet.input, chexnet.layers[-2].output)

checkpoint_filepath = 'C:/Users/Public/mernproject2/server/model_checkpoint.keras'

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

model_1_loaded = tf.keras.models.load_model(checkpoint_filepath, custom_objects={'loss_function': loss_function})

def load_and_preprocess_image(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))  # InceptionV3 expects 299x299 images
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features

def beam(sentence):
    #This function predicts the sentence using beam search
    initial_state = model_1_loaded.layers[0].initialize_states(1)
    encoder_output = model_1_loaded.layers[0](sentence)
    result = ''
    sequences = [['<start>', initial_state, 0]]
    decoder_hidden_state = initial_state
    finished_seq = []
    beam_width = 3
    for i in range(76):
        all_candidates = []
        new_seq = []
        for s in sequences:
            cur_vec = np.reshape(imp2[s[0].split(" ")[-1]], (1, 1))
            decoder_hidden_state = s[1]
            op, hs, attention_weights, context_vector = model_1_loaded.layers[1].onestep(cur_vec, encoder_output, decoder_hidden_state)
            op = tf.nn.softmax(op)
            top3 = np.argsort(op).flatten()[-beam_width:]
            for t in top3:
                candidates = [s[0] + ' ' + imp1[t], hs, s[2] - np.log(np.array(op).flatten()[t])]
                all_candidates.append(candidates)
        sequences = sorted(all_candidates, key=lambda l: l[2])[:beam_width]
        count = 0
        for s1 in sequences:
            if s1[0].split(" ")[-1] == '<end>':
                s1[2] = s1[2] / len(s1[0])  # normalized
                finished_seq.append([s1[0], s1[1], s1[2]])
                count += 1
            else:
                new_seq.append([s1[0], s1[1], s1[2]])
        beam_width -= count
        sequences = new_seq
        if not sequences:
            break
        else:
            continue
    if len(finished_seq) > 0:
        sequences = finished_seq[-1]
        return sequences[0]
    else:
        return new_seq[-1][0]
    


#def predict_caption_for_image(image_path):
    #img_features = load_and_preprocess_image(image_path)
    #prediction = beam(img_features)
    #print("Predicted Sentence is: ", prediction)
    
@app.route('/upload', methods=['POST'])
def upload_image():
    print("image uploaded.....")
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Open the image file
    image = Image.open(io.BytesIO(file.read()))
    print("image is converted into file...")
    # Generate the caption
    #image_features = encode(image).reshape(1,2048)
    #print("image features......")
    #caption = generate_caption( image_features)
    img_features = load_and_preprocess_image(image)
    prediction = beam(img_features)
    print("Predicted Sentence is: ", prediction)
    

    
    return jsonify({'caption': prediction})

if __name__ == '__main__':
    app.run(debug=True)



"""import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications import DenseNet121
from keras.models import load_model
from tensorflow.keras.models import Model


import os
from tensorflow.keras.layers import Input, Softmax, RNN, Dense, Embedding, LSTM,Layer,Dropout,GRU


print("done importing")



@tf.keras.utils.register_keras_serializable()

class Encoder(tf.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns output sequence
    '''

    def _init_(self,units):
        super()._init_()
        self.units = units
        self.dense = Dense(self.units,name = 'Enc_dense')


    def call(self,img):
      '''
          This function takes a sequence input and the initial states of the encoder.
          Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to encoder_lstm
          returns -- All encoder_outputs, last time steps hidden and cell state
      '''
      #enc_out = self.maxpool(tf.expand_dims(img,axis = 2))
      enc_out = self.dense(img)
      return enc_out


    def initialize_states(self,batch_size):
      '''
      Given a batch size it will return intial hidden state and intial cell state.
      If batch size is 32- Hidden state is zeros of size [32,lstm_units], cell state zeros is of size [32,lstm_units]
      '''
      self.batch_size  = batch_size

      self.enc_h =tf.zeros((self.batch_size, self.units))

      #self.enc_c = tf.zeros((self.batch_size, self.lstm_size))
      return self.enc_h

@tf.keras.utils.register_keras_serializable()

class Attention(tf.keras.layers.Layer):
  '''
    Class the calculates score based on the scoring_function using Bahdanu attention mechanism.
  '''
  def _init_(self,att_units):
    # Please go through the reference notebook and research paper to complete the scoring functions
    super()._init_()

    self.att_units = att_units

    self.w1 =  tf.keras.layers.Dense( self.att_units , name = 'w1')
    self.w2 =  tf.keras.layers.Dense( self.att_units,name = 'w2')
    self.v =  tf.keras.layers.Dense(1,name = 'v')

  def call(self,decoder_hidden_state,encoder_output):
    '''
      Attention mechanism takes two inputs current step -- decoder_hidden_state and all the encoder_outputs.
      * Based on the scoring function we will find the score or similarity between decoder_hidden_state and encoder_output.
        Multiply the score function with your encoder_outputs to get the context vector.
        Function returns context vector and attention weights(softmax - scores)
    '''
    self.decoder_hidden_state = decoder_hidden_state
    self.encoder_output = encoder_output


    self.decoder_hidden_state = tf.expand_dims(self.decoder_hidden_state,axis = 1)
    score = self.v(tf.nn.tanh(
              self.w1(self.decoder_hidden_state) + self.w2(self.encoder_output)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * self.encoder_output
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector,attention_weights


@tf.keras.utils.register_keras_serializable()

class OneStepDecoder(tf.keras.Model):
  def _init_(self,tar_vocab_size, embedding_dim, input_length, dec_units  ,att_units):

      # Initialize decoder embedding layer, LSTM and any other objects needed
      super()._init_()
      self.tar_vocab_size = tar_vocab_size
      self.embedding_dim = embedding_dim
      self.input_length = input_length
      self.dec_units = dec_units
      self.att_units = att_units
      self.dec_emb = Embedding(tar_vocab_size,embedding_dim,trainable = True , name = 'dec_embb')
      self.dec_lstm = GRU(self.dec_units, return_state=True, return_sequences=True, name="Decoder_LSTM")
      self.dense   = Dense(self.tar_vocab_size, name = 'one_dec')
      self.attention=Attention( self.att_units)
      self.d1 = Dropout(0.3,name = 'd1')
      self.d2 = Dropout(0.3,name = 'd2')
      self.d3 = Dropout(0.3,name = 'd3')

  @tf.function
  def call(self,input_to_decoder, encoder_output, state_h):
    '''
        One step decoder mechanisim step by step:
      A. Pass the input_to_decoder to the embedding layer and then get the output(batch_size,1,embedding_dim)
      B. Using the encoder_output and decoder hidden state, compute the context vector.
      C. Concat the context vector with the step A output
      D. Pass the Step-C output to LSTM/GRU and get the decoder output and states(hidden and cell state)
      E. Pass the decoder output to dense layer(vocab size) and store the result into output.
      F. Return the states from step D, output from Step E, attention weights from Step -B
    '''
    self.input_to_decoder = input_to_decoder
    self.encoder_output = encoder_output
    self.state_h = state_h

    #A
    target_embedd           = self.dec_emb (self.input_to_decoder)     #(batch_size,1,embedingdim)
    #B
    target_embedd = self.d1(target_embedd)

    context_vector,attention_weights=self.attention(self.state_h,self.encoder_output) #context vector shape = (batch_size,att_units)
    #C
    concated = tf.concat([  tf.expand_dims(context_vector, 1),target_embedd], -1)
    concated = self.d2(concated)

    #D
    lstm_output, hs      = self.dec_lstm(concated, initial_state=self.state_h)

    lstm_output = tf.reshape(lstm_output, (-1, lstm_output.shape[2]))
    lstm_output = self.d3(lstm_output)
    #E
    op = self.dense(lstm_output)
    #op = tf.squeeze(op,[1])
    return op,hs,attention_weights,context_vector

@tf.keras.utils.register_keras_serializable()

class Decoder(tf.keras.Model):
    def _init_(self,out_vocab_size, embedding_dim, input_length, dec_units ,att_units):
      #Intialize necessary variables and create an object from the class onestepdecoder
      super()._init_()
      self.out_vocab_size = out_vocab_size
      self.embedding_dim = embedding_dim
      self.input_length = input_length
      self.dec_units = dec_units
      self.att_units = att_units
      self.onestep = OneStepDecoder(self.out_vocab_size,self.embedding_dim ,self.input_length,self.dec_units,self.att_units)

    @tf.function
    def call(self, input_to_decoder,encoder_output,decoder_hidden_state):


        #Initialize an empty Tensor array, that will store the outputs at each and every time step
        #Create a tensor array as shown in the reference notebook

        #Iterate till the length of the decoder input
            # Call onestepdecoder for each token in decoder_input
            # Store the output in tensorarray
        # Return the tensor array

        all_outputs = tf.TensorArray(tf.float32,size =input_to_decoder.shape[1],name = 'output_arrays' )
        self.input_to_decoder = input_to_decoder
        self.encoder_output = encoder_output
        self.decoder_hidden_state = decoder_hidden_state

        for timestep in tf.range(input_to_decoder.shape[1]):
          op,hs,attention_weights,context_vector = self.onestep(self.input_to_decoder[:,timestep:timestep+1], self.encoder_output, self.decoder_hidden_state)
          self.decoder_hidden_state = hs
          all_outputs = all_outputs.write(timestep,op)
        all_outputs = tf.transpose(all_outputs.stack(),[1,0,2])
        return all_outputs


@tf.keras.utils.register_keras_serializable()
class encoder_decoder(tf.keras.Model):
  #def _init_(self,#params):
    #Intialize objects from encoder decoder
  def _init_(self,out_vocab_size , embedding_size_d, input_length_d,lstm_size_d,att_units,batch_size,units):

        #Create encoder object
        #Create decoder object
        #Intialize Dense layer(out_vocab_size) with activation='softmax'

        super()._init_()

        self.units = units
        self.out_vocab_size = out_vocab_size
        self.embedding_size_d = embedding_size_d
        self.lstm_size_d = lstm_size_d
        self.input_length_d = input_length_d
        self.batch_size = batch_size
        self.att_units = att_units

        self.encoder = Encoder(self.units)

        self.decoder = Decoder(out_vocab_size , embedding_size_d, input_length_d,lstm_size_d,att_units )
        #self.dense   = TimeDistributed(Dense(self.out_vocab_size, activation='softmax'))
        self.dense   = Dense(self.out_vocab_size,name = 'enc_dec_dense')



  def call(self,data):
    #Intialize encoder states, Pass the encoder_sequence to the embedding layer
    # Decoder initial states are encoder final states, Initialize it accordingly
    # Pass the decoder sequence,encoder_output,decoder states to Decoder
    # return the decoder output
    self.inputs,self.outputs = data[0], data[1]
    print("="*20, "ENCODER", "="*20)
    self.encoder_h= self.encoder.initialize_states(self.batch_size)
    self.encoder_output = self.encoder(self.inputs)
    print("-"*27)
    print("ENCODER ==> OUTPUT SHAPE",self.encoder_output.shape)
    print("ENCODER ==> HIDDEN STATE SHAPE",self.encoder_h.shape)
    print("="*20, "DECODER", "="*20)
    output= self.decoder(self.outputs,self.encoder_output,self.encoder_h)
    print("-"*27)
    print("FINAL OUTPUT SHAPE",output.shape)
    print("="*50)
    return output


from tensorflow.keras.preprocessing import image as keras_image
t2 = pd.read_pickle('/home/professor/Downloads/fromgit/Deadline/t2.pickle')

imp1 = {}
imp2 = {}
for key,value in t2.word_index.items():
  imp1[value] = key
  imp2[key] = value




chex_weights = '/home/professor/Downloads/fromgit/archive/chexweights.h5'
chexnet = DenseNet121(weights=chex_weights,                    
                      classes = 14,input_shape=(224,224,3))
model = Model(chexnet.input, chexnet.layers[-2].output)

checkpoint_filepath = '/home/professor/Downloads/fromgit/Deadline/checkpoint/model_checkpoint.keras'

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)
  #out_vocab_size , embedding_size_d, input_length_d,lstm_size_d,att_units,batch_size)
model_1_loaded = tf.keras.models.load_model(checkpoint_filepath, custom_objects={'loss_function': loss_function})


def load_and_preprocess_image(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))  # InceptionV3 expects 299x299 images
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features
def beam(sentence):
    #This function predicts the sentence using beam search
    initial_state = model_1_loaded.layers[0].initialize_states(1)
    encoder_output = model_1_loaded.layers[0](sentence)
    result = ''
    sequences = [['<start>', initial_state, 0]]
    decoder_hidden_state = initial_state
    finished_seq = []
    beam_width = 3
    for i in range(76):
        all_candidates = []
        new_seq = []
        for s in sequences:
            cur_vec = np.reshape(imp2[s[0].split(" ")[-1]], (1, 1))
            decoder_hidden_state = s[1]
            op, hs, attention_weights, context_vector = model_1_loaded.layers[1].onestep(cur_vec, encoder_output, decoder_hidden_state)
            op = tf.nn.softmax(op)
            top3 = np.argsort(op).flatten()[-beam_width:]
            for t in top3:
                candidates = [s[0] + ' ' + imp1[t], hs, s[2] - np.log(np.array(op).flatten()[t])]
                all_candidates.append(candidates)
        sequences = sorted(all_candidates, key=lambda l: l[2])[:beam_width]
        count = 0
        for s1 in sequences:
            if s1[0].split(" ")[-1] == '<end>':
                s1[2] = s1[2] / len(s1[0])  # normalized
                finished_seq.append([s1[0], s1[1], s1[2]])
                count += 1
            else:
                new_seq.append([s1[0], s1[1], s1[2]])
        beam_width -= count
        sequences = new_seq
        if not sequences:
            break
        else:
            continue
    if len(finished_seq) > 0:
        sequences = finished_seq[-1]
        return sequences[0]
    else:
        return new_seq[-1][0]




#this code will predict :-
def predict_caption_for_image(image_path):
    img_features = load_and_preprocess_image(image_path)
    prediction = beam(img_features)
    print("Predicted Sentence is: ", prediction)

# Test the function with an example image path
example_image_path = "/home/professor/Downloads/fromgit/archive/images/augdata/79_IM-2329-2001.dcm.png_aug2.png"  # Replace with the actual image path
predict_caption_for_image(example_image_path)"""