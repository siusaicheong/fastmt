from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import io
import time
import jieba
import json
import keras_preprocessing
from keras_preprocessing import text
#from keras.preprocessing.text import tokenizer_from_json

def view(path, limit=10):
	with open('mini_corpus.txt') as lines:
	  for i,line in enumerate(lines):
	    if i < limit:
	    	print(line)
	    else:
	    	break

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w, lang):
    w = unicode_to_ascii(w.lower().strip())

    if lang == "zh":
    	#z = jieba.lcut(w, cut_all=False)
    	#w = (' ').join(z)
    	w = (' ').join(w)

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    #w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    # w = '<start> ' + w + ' <end>'

    w = '<start> ' + w + ' <end>'

    return w

def create_dataset(src, targ, path, num_examples):
	lang = [src,targ]
	lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
	word_pairs = [[preprocess_sentence(w,lang[i]) for i,w in enumerate(l.split('\t'))]  for l in lines[:num_examples]]
	return zip(*word_pairs)

def max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer

'''
def further_preprocessing(raw, lang):
	output = ()
	if lang == "zh":
		for i,s in enumerate(raw):
			z = jieba.lcut(s, cut_all=False)
			output = (*output, '<start> ' +(' ').join(z)+  ' <end>')
	else:
		for i,e in enumerate(raw):
			output = (*output, '<start> ' + e +  ' <end>')
	return output
'''

def load_dataset(src, targ, path, num_examples=None):
    # creating cleaned input, output pairs
    inp_data, targ_data = create_dataset(src, targ, path, num_examples)
    
    #inp_lang = further_preprocessing(src,src_lang)
    #targ_lang = further_preprocessing(targ,targ_lang)
    print("Creating tokenizer")
    input_tensor, inp_lang_tokenizer = tokenize(inp_data)
    target_tensor, targ_lang_tokenizer = tokenize(targ_data)

    with open('src_tokenizer.json', 'w') as outfile:  
    	json.dump(inp_lang_tokenizer.to_json(), outfile)
    with open('targ_tokenizer.json', 'w') as outfile:  
    	json.dump(targ_lang_tokenizer.to_json(), outfile)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

def loss_function(real, pred,loss_object):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden, encoder, decoder,targ_lang, BATCH_SIZE,loss_object,optimizer):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions,loss_object)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

def evaluate(sentence,lang,checkpoint_dir):
	max_length_targ = 100
	max_length_inp = 100
	units = 1024
	BATCH_SIZE = 64
	embedding_dim = 256

	attention_plot = np.zeros((max_length_targ, max_length_inp))

	sentence = preprocess_sentence(sentence,lang)

	with open('src_tokenizer.json') as json_file:  
		data = json.load(json_file)
		inp_lang = text.tokenizer_from_json(data)
		vocab_inp_size = len(inp_lang.word_index)+1
		encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

		with open('targ_tokenizer.json') as json_file:
		    data = json.load(json_file)
		    targ_lang = text.tokenizer_from_json(data)
		    vocab_tar_size = len(targ_lang.word_index)+1
		    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

		    optimizer = tf.keras.optimizers.Adam()
		    optimizer.beta_1 = 0.9
		    optimizer.beta_2 = 0.999
		    optimizer.decay = 0.0
		    optimizer.learning_rate = 0.001

		    #'''
		    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
	                                 encoder=encoder,
	                                 decoder=decoder)

		    #checkpoint = tf.train.Checkpoint(encoder=encoder,decoder=decoder)

		    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
		    #'''
		    
		    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
		    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
		                                                           maxlen=max_length_inp,
		                                                           padding='post')
		    inputs = tf.convert_to_tensor(inputs)

		    result = ''

		    hidden = [tf.zeros((1, units))]


		    enc_out, enc_hidden = encoder(inputs, hidden)

		    dec_hidden = enc_hidden
		    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

		    
		    for t in range(max_length_targ):

		        predictions, dec_hidden, attention_weights = decoder(dec_input,
		                                                             dec_hidden,
		                                                             enc_out)

		        # storing the attention weights to plot later on
		        attention_weights = tf.reshape(attention_weights, (-1, ))
		        attention_plot[t] = attention_weights.numpy()

		        predicted_id = tf.argmax(predictions[0]).numpy()

		        result += targ_lang.index_word[predicted_id] + ' '

		        if targ_lang.index_word[predicted_id] == '<end>':
		            return result, sentence, attention_plot

		        # the predicted ID is fed back into the model
		        dec_input = tf.expand_dims([predicted_id], 0)

		    return result, sentence, attention_plot
  
# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()
    
def translate(sentence,lang,model='./content/drive/My Drive/training_checkpoints'):
	checkpoint_dir = model
    result, sentence, attention_plot = evaluate(sentence,lang,checkpoint_dir)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    #attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    #plot_attention(attention_plot, sentence.split(' '), result.split(' '))

def get_sample_corpus(lang='cmn'):
	if lang=='eng':
		print("Assuming that you need an English-Chinese corpus")
		lang='cmn'
	path_to_zip = tf.keras.utils.get_file(lang+'eng.zip', origin='http://www.manythings.org/anki/'+lang+'-eng.zip',extract=True)
	path_to_file = os.path.dirname(path_to_zip)+"/"+lang+"-eng/"+lang+".txt"
	print("Corpus saved to the following path: "+path_to_file)
	return path_to_file

def reverse_corpus(path_to_file):
	new_path_to_file = 'rev_'+path_to_file
	with open(path_to_file, "r") as lines:
		with open(new_path_to_file, 'w') as rev_file:
			for line in lines:
				candidates = line.strip().split('\t')
				rev_file.write(('\t').join([candidates[1],candidates[0]])+'\n')
	print("New corpus saved to the following path: "+new_path_to_file)
	return new_path_to_file

def preprocess_data(src, targ, path_to_file='./mini_corpus.txt', num_examples = None):
	
	print("Preprocessing your bilingual data for training. Please wait...")

	input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(src, targ, path_to_file, num_examples)
	max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
	input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

	"""### Create a tf.data dataset"""

	BUFFER_SIZE = len(input_tensor_train)
	BATCH_SIZE = 64
	steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
	
	
	vocab_inp_size = len(inp_lang.word_index)+1
	vocab_tar_size = len(targ_lang.word_index)+1

	dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
	dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

	#example_input_batch, example_target_batch = next(iter(dataset))
	#example_input_batch.shape, example_target_batch.shape

	data = {}
	data['vocab_inp_size'] = vocab_inp_size
	data['vocab_tar_size'] = vocab_tar_size
	data['targ_lang'] = targ_lang #tokenizer
	data['batch_size'] = BATCH_SIZE
	data['dataset'] = dataset
	data['steps_per_epoch'] = steps_per_epoch

	print("Data ready for training!")

	return data

def build_network(data, embedding=256, model='./content/drive/My Drive/training_checkpoints'):
	
	vocab_inp_size = data['vocab_inp_size']
	vocab_tar_size = data['vocab_tar_size']
	targ_lang = data['targ_lang']
	BATCH_SIZE = data['batch_size']

	embedding_dim = embedding
	units = 1024
	checkpoint_dir = model

	encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

	attention_layer = BahdanauAttention(10)
	#attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
	#print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
	#print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

	decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

	'''
	sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)),
	                                      sample_hidden, sample_output)

	print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
	'''

	optimizer = tf.keras.optimizers.Adam()
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

	
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(optimizer=optimizer,
	                                 encoder=encoder,
	                                 decoder=decoder)
	network = {}
	network['encoder'] = encoder
	network['decoder'] = decoder
	network['loss_object'] = loss_object
	network['optimizer'] = optimizer
	network['checkpoint'] = checkpoint
	network['checkpoint_prefix'] = checkpoint_prefix

	return network

def train_model(data,network,EPOCHS=10,save_freq=2):
	save_freq=max(int(save_freq),1)
	targ_lang = data['targ_lang'] #tokenizer
	BATCH_SIZE = data['batch_size']
	dataset = data['dataset']
	steps_per_epoch = data['steps_per_epoch']
	encoder = network['encoder']
	decoder = network['decoder']
	loss_object = network['loss_object']
	optimizer = network['optimizer']
	checkpoint = network['checkpoint']
	checkpoint_prefix = network['checkpoint_prefix']

	print("Start training...")
	print()

	for epoch in range(EPOCHS):
	  start = time.time()

	  enc_hidden = encoder.initialize_hidden_state()
	  total_loss = 0

	  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
	    batch_loss = train_step(inp, targ, enc_hidden, encoder, decoder, targ_lang, BATCH_SIZE,loss_object,optimizer)
	    total_loss += batch_loss
	    
	    if batch % 50 == 0:
	      print('Epoch {} Batch {}'.format(epoch + 1,batch))

	    if batch >0 and batch % 100 == 0:
	        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
	                                                     batch,
	                                                     batch_loss.numpy()))
	  # saving (checkpoint) the model every 2 epochs
	  if (epoch + 1) % save_freq == 0:
	    checkpoint.save(file_prefix = checkpoint_prefix)

	  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
	                                      total_loss / steps_per_epoch))
	  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

	print("Training completed.")



def train(src, targ, path_to_file='./mini_corpus.txt',num_examples=20000,checkpoint_dir = './content/drive/My Drive/training_checkpoints',EPOCHS = 10):
	#path_to_file = '/content/mini_corpus.txt'
	#num_examples = 20000

	print("Step 1: Preprocessing your bilingual data for training. Please wait...")

	input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(src, targ, path_to_file, num_examples)
	max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
	input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

	"""### Create a tf.data dataset"""

	BUFFER_SIZE = len(input_tensor_train)
	BATCH_SIZE = 64
	steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
	embedding_dim = 256
	units = 1024
	vocab_inp_size = len(inp_lang.word_index)+1
	vocab_tar_size = len(targ_lang.word_index)+1

	dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
	dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

	example_input_batch, example_target_batch = next(iter(dataset))
	example_input_batch.shape, example_target_batch.shape

	print("Step 2: Defining the neural network...")

	encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

	attention_layer = BahdanauAttention(10)
	#attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
	#print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
	#print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

	decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

	'''
	sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)),
	                                      sample_hidden, sample_output)

	print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
	'''

	optimizer = tf.keras.optimizers.Adam()
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

	
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(optimizer=optimizer,
	                                 encoder=encoder,
	                                 decoder=decoder)
	

	print("Step 3: Start training...")
	print()

	for epoch in range(EPOCHS):
	  start = time.time()

	  enc_hidden = encoder.initialize_hidden_state()
	  total_loss = 0

	  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
	    batch_loss = train_step(inp, targ, enc_hidden, encoder, decoder, targ_lang, BATCH_SIZE,loss_object,optimizer)
	    total_loss += batch_loss
	    
	    if batch % 50 == 0:
	      print('Epoch {} Batch {}'.format(epoch + 1,batch))

	    if batch >0 and batch % 100 == 0:
	        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
	                                                     batch,
	                                                     batch_loss.numpy()))
	  # saving (checkpoint) the model every 2 epochs
	  if (epoch + 1) % 2 == 0:
	    checkpoint.save(file_prefix = checkpoint_prefix)

	  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
	                                      total_loss / steps_per_epoch))
	  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

	print("Training completed.")