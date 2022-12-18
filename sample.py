"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PY GAME

https://pygame-learning-environment.readthedocs.io/en/latest/user/games/flappybird.html

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
from os.path import exists

import tensorflow as tf

import ple
from ple import PLE

from ple.games.flappybird import FlappyBird as flappybird_game
from ple.games import base
from pygame.constants import K_w, K_h

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
None
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
actions = { "none_1": K_h, "up_1": K_w, "none_2": K_h, "none_3": K_h, "none_4": K_h, "none_5": K_h, "none_6": K_h, "none_7": K_h, "none_8": K_h, "none_9": K_h }
nb_frames = 100000000000
global reward
reward = 0.0
global step
step = 0
global gamescores
gamescores = 0.0
##
globalstep = 0
game_global_step = 0
global action
action = 0
global gameState
gameState = {'player_y': 256, 'player_vel': 0, 'next_pipe_dist_to_player': 309.0, 'next_pipe_top_y': 97, 'next_pipe_bottom_y': 347, 'next_next_pipe_dist_to_player': 453.0, 
'next_next_pipe_top_y': 113, 'next_next_pipe_bottom_y': 363}

global DATA
DATA = tf.zeros([1, 1, 1, 30], dtype=tf.float32)
global LABEL
LABEL = tf.zeros([1, 1, 1, 1], dtype=tf.float32)

### Mixed of data input
for i in range(15):
	DATA_row = tf.constant([ 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999,
				-9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999 ], shape=(1, 1, 1, 30), dtype=tf.float32)		
	DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
	LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(0, shape=(1, 1, 1, 1))])
	
for i in range(15):
	DATA_row = tf.constant([ -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999,
				9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999 ], shape=(1, 1, 1, 30), dtype=tf.float32)		
	DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
	LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(9, shape=(1, 1, 1, 1))])	
	
DATA = DATA[-30:,:,:,:]
LABEL = LABEL[-30:,:,:,:]
			
momentum = 0.1

learning_rate = 0.0001
batch_size=10

checkpoint_path = "F:\\models\\checkpoint\\" + os.path.basename(__file__).split('.')[0] + "\\TF_DataSets_01.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

if not exists(checkpoint_dir) : 
	os.mkdir(checkpoint_dir)
	print("Create directory: " + checkpoint_dir)
	
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Environment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
game_console = flappybird_game(width=288, height=512, pipe_gap=250)		# pipe_gap=100	# pipe_gap=250	# pipe_gap=220	# pipe_gap=180 # pipe_gap=80
p = PLE(game_console, fps=30, display_screen=True)
p.init()

obs = p.getScreenRGB()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def	read_current_sate( string_gamestate ):
	global gameState


	gameState = p.getGameState()
	
	
	
	if string_gamestate in ['player_y', 'player_vel', 'next_pipe_dist_to_player', 'next_pipe_top_y', 'next_pipe_bottom_y', 'next_next_pipe_dist_to_player', 'next_next_pipe_top_y', 'next_next_pipe_bottom_y']:
		return round( gameState[string_gamestate], 1 )
	elif string_gamestate == 'near_gap':
		return round( gameState['next_pipe_bottom_y'] - (( gameState['next_pipe_bottom_y'] - gameState['next_pipe_top_y'] ) / 2) - gameState['player_y'], 1 )
	elif string_gamestate == 'far_gap':
		return round( gameState['next_next_pipe_bottom_y'] - (( gameState['next_next_pipe_bottom_y'] - gameState['next_next_pipe_top_y'] ) / 2) - gameState['player_y'], 1 )
	elif string_gamestate == 'acceleration':
		return round( gameState['player_vel'] * gameState['next_pipe_dist_to_player'], 1 )
	elif string_gamestate == 'distance_velocity':
		return round( gameState['player_y'] * gameState['player_vel'] * gameState['next_pipe_dist_to_player'], 1 )
	else:
		return None
		
def predict_action( ):
	global DATA
	
	predictions = model.predict(tf.expand_dims(tf.squeeze(DATA), axis=1 ))
	score = tf.nn.softmax(predictions[0])

	return int(tf.math.argmax(score))
	
def update_DATA( action ):
	global gameState
	global reward
	global step
	global gamescores
	global DATA
	global LABEL
	
	step = step + 1
	
	gameState = p.getGameState()
	player_y_array = gameState['player_y']
	player_vel_array = gameState['player_vel']
	next_pipe_dist_to_player_array = gameState['next_pipe_dist_to_player']
	next_pipe_top_y_array = gameState['next_pipe_top_y']
	next_pipe_bottom_y_array = gameState['next_pipe_bottom_y']
	next_next_pipe_dist_to_player_array = gameState['next_next_pipe_dist_to_player']
	next_next_pipe_top_y_array = gameState['next_next_pipe_top_y']
	next_next_pipe_bottom_y_array = gameState['next_next_pipe_bottom_y']
		
	######################################################################################
	gap = (( next_pipe_bottom_y_array - next_pipe_top_y_array ) / 2 )
	top = next_pipe_top_y_array
	target = top + gap
	
	height_diff = player_y_array - next_pipe_top_y_array
	angle_diff = height_diff / ( next_pipe_dist_to_player_array + 1 )
	
	height_diff_2 = next_pipe_top_y_array - next_next_pipe_top_y_array
	angle_diff_2 = height_diff_2 / ( next_next_pipe_dist_to_player_array - next_pipe_dist_to_player_array + 1 )
	
	height_diff_3 = player_y_array - next_next_pipe_top_y_array
	angle_diff_3 = height_diff_3 / ( next_pipe_dist_to_player_array + 1 )
	
	height_diff_4 = next_pipe_top_y_array - next_next_pipe_bottom_y_array
	angle_diff_4 = height_diff_4 / ( next_next_pipe_dist_to_player_array - next_pipe_dist_to_player_array + 1 )
	
	angle_diff_5 = angle_diff_2 * angle_diff_3 / ( angle_diff + 1 )
	
	
	coefficient_0 = next_pipe_bottom_y_array - player_y_array
	coefficient_0 = angle_diff_4 * ( coefficient_0 - (player_y_array - target) )
	
	coefficient_0 = next_pipe_bottom_y_array - player_y_array + 0
	coefficient_1 = next_pipe_bottom_y_array - player_y_array + 5
	coefficient_2 = next_pipe_bottom_y_array - player_y_array + 10
	coefficient_3 = next_pipe_bottom_y_array - player_y_array + 30
	coefficient_4 = next_pipe_bottom_y_array - player_y_array + 40
	coefficient_5 = player_y_array - next_pipe_top_y_array + 0
	coefficient_6 = player_y_array - next_pipe_top_y_array + 5
	coefficient_7 = player_y_array - next_pipe_top_y_array + 10
	coefficient_8 = player_y_array - next_pipe_top_y_array + 15
	coefficient_9 = player_y_array - next_pipe_top_y_array + 20
	
	coefficient_0 = coefficient_0 - (player_y_array - target)
	coefficient_1 = coefficient_1 - (player_y_array - target)
	coefficient_2 = coefficient_2 - (player_y_array - target)
	coefficient_3 = coefficient_3 - (player_y_array - target)
	coefficient_4 = coefficient_4 - (player_y_array - target)
	coefficient_5 = coefficient_5 - (player_y_array - target)
	coefficient_6 = coefficient_6 - (player_y_array - target)
	coefficient_7 = coefficient_7 - (player_y_array - target)
	coefficient_8 = coefficient_8 - (player_y_array - target)
	coefficient_9 = coefficient_9 - (player_y_array - target)
	######################################################################################

	contrl = ( player_y_array - next_pipe_dist_to_player_array ) + ( 50 * reward )
	coff_0 = coefficient_0
	coff_1 = coefficient_5
	coff_2 = 1
	coff_3 = 1
	coff_4 = 1
	coff_5 = 1
	coff_6 = 1
	coff_7 = 1
	coff_8 = 1
	coff_9 = 1
	coff_10 = 1
	coff_11 = 1
	
	print( "step: " + str( step ).zfill(6) + " action: " + str(int(action)).zfill(1) + " coff_0: " + str(int(coff_0)).zfill(6) + " coff_1: " + str(int(coff_1)).zfill(6) + " coff_2: " 
			+ str(int(coff_2)).zfill(6) + " coff_3: " + str(int(coff_3)).zfill(6) + " coff_4: " + str(int(coff_4)).zfill(6)
	)
	
	DATA_row = tf.constant([ contrl, coff_0, coff_1, coff_2, coff_3, coff_4, coff_5, coff_6, coff_7, coff_8, coff_9, coff_10, coff_11,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ], shape=(1, 1, 1, 30), dtype=tf.float32)
	
	DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
	DATA = DATA[-30:,:,:,:]
	
	LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(action, shape=(1, 1, 1, 1))])
	LABEL = LABEL[-30:,:,:,:]
	
	DATA = DATA[-30:,:,:,:]
	LABEL = LABEL[-30:,:,:,:]
	
	return DATA, LABEL, step

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Callback
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class custom_callback(tf.keras.callbacks.Callback):

	def __init__(self, patience=0):
		self.best_weights = None
		self.best = 999999999999999
		self.patience = patience
	
	def on_train_begin(self, logs={}):
		self.best = 999999999999999
		self.wait = 0
		self.stopped_epoch = 0

	def on_epoch_end(self, epoch, logs={}):
		if(logs['accuracy'] == None) : 
			pass
		
		if logs['loss'] < self.best :
			self.best = logs['loss']
			self.wait = 0
			self.best_weights = self.model.get_weights()
		else :
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				print("Restoring model weights from the end of the best epoch.")
				self.model.set_weights(self.best_weights)

		if self.wait > self.patience :
			self.model.stop_training = True

custom_callback = custom_callback(patience=8)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: DataSet
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dataset = tf.data.Dataset.from_tensor_slices((DATA, LABEL))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Initialize
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=(1, 30)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, return_state=False)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
	tf.keras.layers.Dense(256),
	tf.keras.layers.Dropout(.4, input_shape=(256,))
])
		
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(192))
model.add(tf.keras.layers.Dense(10))
model.summary()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Optimizer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=momentum,
    nesterov=False,
    name='SGD',
)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Loss Fn
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""								
lossfn = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.AUTO, name='mean_squared_logarithmic_error')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Summary
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model.compile(optimizer=optimizer, loss=lossfn, metrics=['accuracy'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: FileWriter
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if exists(checkpoint_path) :
	model.load_weights(checkpoint_path)
	print("model load: " + checkpoint_path)
	input("Press Any Key!")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Training
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
history = model.fit(dataset, epochs=1, callbacks=[custom_callback])
model.save_weights(checkpoint_path)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Tasks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
for i in range(nb_frames):
	game_global_step = game_global_step + 1
	gamescores = gamescores + reward
	reward = 0
	
	DATA, LABEL, step = update_DATA( action )
	
	if p.game_over():	
		step = 0
		gamescores = 0
		reward = 0

		p.init()
		p.reset_game()
	
		DATA, LABEL, step = update_DATA( action )
	
	action = predict_action( )
	reward = p.act(list(actions.values())[action])
	
	if ( reward > 0 and step > 0  ):
		dataset = tf.data.Dataset.from_tensor_slices((DATA, LABEL))
		history = model.fit(dataset, epochs=2, batch_size=batch_size, callbacks=[custom_callback])
		model.save_weights(checkpoint_path)
	
	if ( reward != 0 and step > 0  ):
		dataset = tf.data.Dataset.from_tensor_slices((DATA, LABEL))
		history = model.fit(dataset, epochs=2, batch_size=batch_size, callbacks=[custom_callback])
		model.save_weights(checkpoint_path)
