# AI_game_auto_play
AI games auto-play networks training, linear regression, everything in control linear adding, subtraction and ( 50 * reward value ). It proved that simple tasks are done by simple approach to the solution by adding summation, that create SGD optimizer is making senses with linear regression even loss function is root mean square.

##### 👨🏻‍🏫💬 Positive ######

1. It is self control equation, ( distance_to_goal - player_y_array ) is enough for running to goal when adding more dismension ( player_x_array - object_y_array ) is enough for apprache few jumps on target too.
2. All varaiables is consider do not use 0 since it is significant and make anybody significant or you try 13 or 73
3. Our game object is pass though each gap as possible with time efficientcy or rewards feed.
4. Power, Square, Absolute, Devide, Centimeters and applied it will make the networks learn fast but it hard to control when there has some problem and your equation start more complex for complete all tasks may may need time or more samples required.

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

Only LSTM layer is enough but I added Dense layer because the last model had problem with overfilt it play all night but few day it become overfits even I added exception in traning callbacks. 👨🏻‍🏫💬 Dense layer transfrom accumulate works to output different, over fitting problems alway found in AI games problem try to create limits and stop before it overcomes you. 
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

I select simple optimizer that we can estimate of it work as simple before you can apply your optimizer that is because you can assume the loss function value and it make sense when actions learning display as list start changing as this example { 1, 1, 1, 1, 1, 9, 9, 9 ,9 ,9 },  { 1, 1, 1, 1, 5, 9, 9, 9 ,9 ,9 },  { 1, 1, 7, 4, 5, 9, 9, 9 ,9 ,9 } ...

The SGD is simple as ( New target = AX + BY + C ) 💃( 👩‍🏫 )💬 The demads graph is always increase by time to meet the satisfaction revese it becomes ( BY - AX + C ) because you using previous data record and when have different of demand area you know the value in your stocks that help.
```
optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=momentum,
    nesterov=False,
    name='SGD',
)
```

## Loss Function ##

Loss function approches is in logarithm scales then you don't need to compare all long history with some point in the median but it repeating tasks and report of the running accuracy as linear algorithms. 👨🏻‍🏫💬 Using linear regression you don't need to repeating the same task and logarithm preserved some properties of networks call transmissions.
```
lossfn = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.AUTO, 
                  name='mean_squared_logarithmic_error')
```
## Taget actions ##

For accuracy action I leaved none action at the begining and end, I added some none action between to confrim the networks accuracy it will make you training harder or sometimes it required few time relarning from start but surely your network if success it will be accuracy networks. 👩‍🏫💬 You setup first action at the most right or left distributions of the networks is working very fast, plots graph outside of the area will consider as action[0] than action[1] then it hit the ceiling often. 
```
actions = { "none_1": K_h, "up_1": K_w, "none_2": K_h, "none_3": K_h, "none_4": K_h, 
                      "none_5": K_h, "none_6": K_h, "none_7": K_h, "none_8": K_h, "none_9": K_h }
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
