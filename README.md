# AI_game_auto_play
AI games auto-play networks training, linear regression

```
coefficient_0 = next_pipe_bottom_y_array - player_y_array + 0
coefficient_0 = coefficient_0 - (player_y_array - target)

coefficient_5 = player_y_array - next_pipe_top_y_array + 0
coefficient_5 = coefficient_5 - (player_y_array - target)

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

DATA_row = tf.constant([ contrl, coff_0, coff_1, coff_2, coff_3, coff_4, coff_5, coff_6, coff_7, coff_8, coff_9, coff_10, coff_11,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ], shape=(1, 1, 1, 30), dtype=tf.float32)
```

## Model ##

```
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
```

## Optimizer ##

```
optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=momentum,
    nesterov=False,
    name='SGD',
)
```

## Loss Function ##

```
lossfn = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.AUTO, 
                  name='mean_squared_logarithmic_error')
```

## Files and Directory ##

Files and directory perfrom in the action, training and play without external input proves our concept it true.
1. sample.py : sample Python script to train networks for AI auto-play. 
2. 10.png : screen shot indicates AI self-learning from environment.
3. FlappyBird_small.gif : screen shot indicates AI game actually play in the game environment.
4. README.md : readme file.

## Result ##

This screen shot indicate AI auto-play is fully learning by itself, not use guiding I create and test both methods see previous example.
![Alt text](https://github.com/jkaewprateep/AI_game_auto_play/blob/main/10.png?raw=true "Title")

This Flappy birds game is good example because no sample action cheats and accuracy actions.
![Alt text](https://github.com/jkaewprateep/AI_game_auto_play/blob/main/FlappyBird_small.gif?raw=true "Title")
