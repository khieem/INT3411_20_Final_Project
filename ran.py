import pygame
import time
import random
import threading
import sys
import os
import sounddevice as sd
from scipy.io.wavfile import write
from pynput.keyboard import Listener
import librosa
import torch 
import torch.nn as nn

cmd = 'RIGHT' # DOWN LEFT RIGHT
in_use = 0
semaphore = threading.Condition()

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes, device, classes=None):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)
		self.device = device
		self.classes = classes

	def forward(self, x):
		batch_size = x.size(0)
		h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) 
		c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) 
	
		out, _ = self.lstm(x, (h0, c0))  # shape = (batch_size, seq_length, hidden_size)
	 
	 # Decode the hidden state of the last time step
		out = self.fc(out[:, -1, :])
		return out

def load_model():
	map_location=torch.device('cpu')
	ver = torch.load("./ver2.pth",map_location=torch.device('cpu'))
	model = RNN(input_size=20, hidden_size=64, num_layers=3, num_classes=4, device="cpu")
	#  print(model)
	model.load_state_dict(ver["model"])
	return model

def rec():
	t = 1.5
	n = 0
	sr = 22050
  
	print('ghi...')
	raw_rec = sd.rec(int(t*sr), samplerate=sr, channels=1)
	rec = raw_rec.reshape((-1, ))
	# rec = raw_rec
	sd.wait()
	# write('out{}.wav'.format(n), sr, rec)
	return rec # rec nhét thẳng vào librosa. raw_rec để ghi ra .wav
import numpy as np
def normalize(feature):
  normalized = np.full_like(feature, 0)
  for i in range(feature.shape[1]):
    normalized[:,i] = feature[:,i] - np.mean(feature[:,i]) # đưa trung bình về 0
    normalized[:,i] = normalized[:,i] / np.max(np.abs(normalized[:,i])) # đưa khoảng giá trị về [-1, 1]
  return normalized

def regconize():
	signal = rec()
	clip = librosa.effects.trim(signal, top_db= 10)
	#  print(signal.shape, clip[0].shape)
	#  print("Bắt đầu dự đoán")
	 
	mfccs = normalize(librosa.feature.mfcc(
				y=clip[0],
				n_mfcc=20,
		  ))
	commands = ["LEFT","RIGHT","UP", "DOWN"]
	labels = ["trai","phai","len", "xuong"]
	 
	model = load_model()    
	input = torch.Tensor(mfccs.T)
	input = input.unsqueeze(0)
	pred = model(input)
	#  pred_label = torch.max(pred, dim=1)
	#  pred_label = torch.max(pred, dim=1)
	#  print(pred_label)
	#  label = all_labels[pred_label[1].numpy()[0]]
	cmd = commands[torch.argmax(pred)]
	label = labels[torch.argmax(pred)]
	print(label)
	return cmd

def game():
	snake_speed = 6
	fruit_size = 50
	window_x = 720
	window_y = 480

	black = pygame.Color(0, 0, 0)
	white = pygame.Color(255, 255, 255)
	red = pygame.Color(255, 0, 0)
	green = pygame.Color(0, 255, 0)

	pygame.init()
	pygame.display.set_caption('ranvjpro')
	game_window = pygame.display.set_mode((window_x, window_y))

	fps = pygame.time.Clock()

	snake_position = [100, 70]

	snake_body = [[100, 70], [90, 70], [80, 70], [70, 70]]

	fruit_position = [random.randrange(1, (window_x//fruit_size)) * fruit_size,
					random.randrange(1, (window_y//fruit_size)) * fruit_size]

	fruit_spawn = True

	# setting default snake direction towards
	# right
	direction = 'DOWN'
	change_to = direction

	# initial score
	score = 0

	def show_score(choice, color, font, size):
		score_font = pygame.font.SysFont(font, size)
		score_surface = score_font.render('Điểm : ' + str(score), True, color)
		score_rect = score_surface.get_rect()
		game_window.blit(score_surface, score_rect)

	def game_over():
		my_font = pygame.font.SysFont('times new roman', 50)
		game_over_surface = my_font.render(
			'Điểm : ' + str(score), True, red)
		game_over_rect = game_over_surface.get_rect()
		game_over_rect.midtop = (window_x/2, window_y/4)
		game_window.blit(game_over_surface, game_over_rect)
		pygame.display.flip()
		time.sleep(1)
		pygame.quit()
		os._exit(0)


	global cmd
	global in_use
	while True:      
		for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					os._exit(0)
		if in_use == 1:
			semaphore.acquire()
		if cmd == 'UP':
			change_to = 'UP'
		elif cmd == 'DOWN':
			change_to = 'DOWN'
		elif cmd == 'LEFT':
			change_to = 'LEFT'
		elif cmd == 'RIGHT':
			change_to = 'RIGHT'
		if in_use == 1:
			in_use = 0
			semaphore.wait()
		# else:
		#    semaphore.wait()
		if change_to == 'UP' and direction != 'DOWN':
			direction = 'UP'
		if change_to == 'DOWN' and direction != 'UP':
			direction = 'DOWN'
		if change_to == 'LEFT' and direction != 'RIGHT':
			direction = 'LEFT'
		if change_to == 'RIGHT' and direction != 'LEFT':
			direction = 'RIGHT'

		if direction == 'UP':
			snake_position[1] -= 10
		if direction == 'DOWN':
			snake_position[1] += 10
		if direction == 'LEFT':
			snake_position[0] -= 10
		if direction == 'RIGHT':
			snake_position[0] += 10

		snake_body.insert(0, list(snake_position))
		if snake_position[1] >= fruit_position[1] and snake_position[1] <= fruit_position[1] + fruit_size:
			if snake_position[0] + 10 >= fruit_position[0] and snake_position[0] <= fruit_position[0] + fruit_size:
				score += 20
				fruit_spawn = False
			else:
				snake_body.pop()
		elif snake_position[0] >= fruit_position[0] and snake_position[0] <= fruit_position[0] + fruit_size:
			if snake_position[1] + 10 >= fruit_position[1] and snake_position[1] <= fruit_position[1] + fruit_size:
				score += 20
				fruit_spawn = False
			else:
				snake_body.pop()
		else:
			snake_body.pop()
			
		if not fruit_spawn:
			fruit_position = [random.randrange(1, (window_x//fruit_size)) * fruit_size,
							random.randrange(1, (window_y//fruit_size)) * fruit_size]
			
		fruit_spawn = True
		game_window.fill(black)
		
		for pos in snake_body:
			pygame.draw.rect(game_window, green,
							pygame.Rect(pos[0], pos[1], 10, 10))
		pygame.draw.rect(game_window, white, pygame.Rect(
			fruit_position[0], fruit_position[1], fruit_size, fruit_size))

		# Game Over conditions
		if snake_position[0] < 0 or snake_position[0] > window_x-10:
			game_over()
		if snake_position[1] < 0 or snake_position[1] > window_y-10:
			game_over()

		# can duoi
		for block in snake_body[1:]:
			if snake_position[0] == block[0] and snake_position[1] == block[1]:
				game_over()

		show_score(1, white, 'times new roman', 20)
		pygame.display.update()
		fps.tick(snake_speed)



def getKey(key):
	global cmd
	global in_use
	key = str(key)
	print(key)
	
	if in_use == 1:
		semaphore.release()
	if key != 'Key.enter':
		cmd = regconize()
	# if key == 'Key.down':
	#    cmd = 'DOWN'
	# elif key == 'Key.up':
	#    cmd = 'UP'
	# elif key == 'Key.left':
	#    cmd = 'LEFT'
	# elif key == 'Key.right':
	#    cmd = 'RIGHT'
	# elif key == '\'r\'':
	#    rec()
	if in_use == 1:
		in_use = 0
		semaphore.wait()

if __name__ == '__main__':
	p_game = threading.Thread(target=game)
	try:
		p_game.start()
		with Listener(on_press=getKey) as listener:
			listener.join()
	except (KeyboardInterrupt, SystemExit):
		sys.exit()
	# p_rec = threading.Thread(target=rec)
	# p_rec.start()
	# p_game.join()
	# p_rec.join()
