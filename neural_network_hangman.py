import tkinter as tk
from tkinter import ttk
import random
import csv
import os
import time
import threading
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, Counter

# Optional torch import
try:
	import torch
	import torch.nn as nn
except Exception:
	torch = None
	nn = None

# Function to load words from the provided wordbank
def load_words_from_csv(file_path):
	words = []
	
	try:
		with open(file_path, newline='', encoding='utf-8') as csvfile:
			words = [word.strip().lower() for word in csvfile.read().split(',') if word.strip()]
	except FileNotFoundError:
		print(f"Word bank file not found at {file_path}")
		words = ['python', 'programming', 'computer', 'hangman', 'game', 'artificial', 'intelligence', 'machine', 'learning']
	
	return words

# Load words from the provided wordbank
words = load_words_from_csv('Hangman_wordbank')

class GameState:
	"""Represents the current state of the Hangman game for AI players"""
	def __init__(self, word: str, max_attempts: int = 9):
		self.word = word
		self.max_attempts = max_attempts
		self.attempts_left = max_attempts
		self.guessed_letters = set()
		self.correct_letters = set()
		self.incorrect_letters = set()
		self.game_over = False
		self.won = False
		
	def get_pattern(self) -> str:
		"""Get the current word pattern (e.g., 'p _ t h o n')"""
		return ' '.join([letter if letter in self.correct_letters else '_' for letter in self.word])
	
	def get_available_letters(self) -> List[str]:
		"""Get letters that haven't been guessed yet"""
		return [chr(i) for i in range(ord('a'), ord('z') + 1) if chr(i) not in self.guessed_letters]
	
	def make_guess(self, letter: str) -> Tuple[bool, int]:
		"""
		Make a guess and return (is_correct, reward)
		Reward: +1 for correct, -1 for wrong, +10 for win, -10 for loss
		"""
		if letter in self.guessed_letters or self.game_over:
			return False, 0
			
		self.guessed_letters.add(letter)
		
		if letter in self.word:
			self.correct_letters.add(letter)
			self.attempts_left += 1  # Correct guesses don't count against attempts
			reward = 1
			
			# Check for win
			if all(letter in self.correct_letters for letter in self.word):
				self.won = True
				self.game_over = True
				reward = 10
		else:
			self.incorrect_letters.add(letter)
			self.attempts_left -= 1
			reward = -1
			
			# Check for loss
			if self.attempts_left <= 0:
				self.game_over = True
				reward = -10
				
		return letter in self.word, reward
	
	def is_game_over(self) -> bool:
		return self.game_over

class TorchLetterPredictor(nn.Module if nn is not None else object):
	def __init__(self, input_size: int, hidden_size: int = 512, num_classes: int = 26):
		if nn is not None:
			super().__init__()
			self.net = nn.Sequential(
				nn.Linear(input_size, hidden_size),
				nn.ReLU(),
				nn.Linear(hidden_size, hidden_size),
				nn.ReLU(),
				nn.Linear(hidden_size, num_classes),
			)
		else:
			pass
	def forward(self, x):
		return self.net(x)

class NeuralNetworkAI:
	"""Advanced Neural Network AI using pattern recognition and optional NN model"""
	
	def __init__(self, model_path: str = 'models/letter_predictor.pt', max_len: int = 20):
		self.letter_to_index = {chr(i): i - ord('a') for i in range(ord('a'), ord('z') + 1)}
		self.word_patterns = {}
		self.letter_probabilities = {}
		self.learning_data = []
		self.model_path = model_path
		self.max_len = max_len
		self.device = None
		self.model = None
		self._maybe_load_model()
	
	def _maybe_load_model(self):
		if torch is None:
			return
		if not os.path.exists(self.model_path):
			return
		ckpt = torch.load(self.model_path, map_location='cpu')
		cfg = ckpt.get('config', {})
		input_size = cfg.get('input_size', (self.max_len) * (26+1))
		self.device = torch.device('cuda' if torch and torch.cuda.is_available() else 'cpu')
		self.model = TorchLetterPredictor(input_size)
		self.model.load_state_dict(ckpt['state_dict'])
		self.model.to(self.device)
		self.model.eval()
	
	def _encode_pattern(self, pattern: str):
		# One-hot over 27 symbols per position (26 letters + underscore)
		symbols = ['_'] + [chr(i) for i in range(ord('a'), ord('z')+1)]
		sym_to_idx = {s:i for i,s in enumerate(symbols)}
		max_len = self.max_len
		import torch as _torch  # local alias
		encoded = _torch.zeros(max_len, len(symbols), dtype=_torch.float32)
		p = pattern[:max_len].ljust(max_len, '_')
		for i,ch in enumerate(p):
			encoded[i, sym_to_idx.get(ch, 0)] = 1.0
		return encoded.view(-1)
	
	def get_guess(self, game_state: GameState) -> str:
		"""Get next guess using model if available, else heuristics"""
		available_letters = game_state.get_available_letters()
		if not available_letters:
			return 'a'
		
		pattern = game_state.get_pattern().replace(' ', '')
		
		# If we have a trained model, use it to rank letters
		if self.model is not None and torch is not None:
			with torch.no_grad():
				x = self._encode_pattern(pattern).unsqueeze(0).to(self.device)
				logits = self.model(x)
				probs = torch.softmax(logits, dim=1).squeeze(0)
				# Sort letters by probability
				letter_scores = [
					(chr(ord('a')+i), float(probs[i].item())) for i in range(26)
				]
				letter_scores.sort(key=lambda t: t[1], reverse=True)
				for letter, _ in letter_scores:
					if letter in available_letters:
						return letter
		
		# Fallback to learned patterns
		if pattern in self.word_patterns:
			best_letters = self.word_patterns[pattern]
			for letter in best_letters:
				if letter in available_letters:
					return letter
		
		# Heuristic pattern analysis
		pattern_letters = self.analyze_pattern(pattern)
		for letter in pattern_letters:
			if letter in available_letters:
				return letter
		
		# Common letters fallback
		common_letters = ['e', 'a', 'r', 'i', 'o', 't', 'n', 's', 'l', 'c']
		for letter in common_letters:
			if letter in available_letters:
				return letter
		
		return random.choice(available_letters)
	
	def analyze_pattern(self, pattern: str) -> List[str]:
		suggestions = []
		word_length = len(pattern)
		missing_count = pattern.count('_')
		if missing_count == 0:
			return suggestions
		if word_length >= 3:
			if pattern[0] == '_' and pattern[-1] != '_':
				if word_length <= 4:
					suggestions.extend(['t', 'a', 's', 'c', 'm', 'b', 'f', 'h'])
				else:
					suggestions.extend(['t', 'a', 's', 'c', 'm', 'p', 'r', 'd'])
			elif pattern[-1] == '_' and pattern[0] != '_':
				suggestions.extend(['e', 's', 'd', 'n', 'r', 't', 'y', 'g'])
			elif '_' in pattern[1:-1]:
				suggestions.extend(['e', 'a', 'i', 'o', 'u'])
				suggestions.extend(['r', 'n', 't', 'l', 's'])
		consonants = set(pattern) - {'_'}
		if len(consonants) > missing_count and len(suggestions) < 3:
			suggestions.extend(['e', 'a', 'i', 'o', 'u'])
		return suggestions
	
	def learn_from_game(self, pattern: str, correct_letters: List[str], won: bool):
		if won and correct_letters:
			if pattern not in self.word_patterns:
				self.word_patterns[pattern] = []
			self.word_patterns[pattern].extend(correct_letters)
			self.learning_data.append({'pattern': pattern, 'correct_letters': correct_letters, 'won': won})
			if len(self.learning_data) > 100:
				self.learning_data = self.learning_data[-100:]
	
	def reset(self):
		pass

class NeuralNetworkHangmanGame:
	def __init__(self, window):
		self.window = window
		self.window.title("Neural Network AI Hangman")
		self.window.config(bg="lightblue")
		self.window.geometry("800x600")

		# Game variables
		self.words = words.copy()
		self.selected_word = ""
		self.game_state = None
		self.game_over = False
		self.ai = NeuralNetworkAI()
		self.ai_stats = {"wins": 0, "losses": 0, "total_games": 0, "learning_curve": []}
		self.ai_playing = False
		
		# Create UI elements
		self.create_ui()
		
		# Start first game
		self.start_new_game()

	def create_ui(self):
		# Main frame
		main_frame = tk.Frame(self.window, bg="lightblue")
		main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
		
		# Title
		title_label = tk.Label(main_frame, text="Neural Network AI Hangman", font=('Helvetica', 24, 'bold'), bg="lightblue")
		title_label.pack(pady=20)
		
		# Word display frame (visible to human)
		word_frame = tk.LabelFrame(main_frame, text="Word to Guess (Visible to Human)", font=('Helvetica', 14, 'bold'), bg="lightblue")
		word_frame.pack(fill=tk.X, pady=10)
		
		self.word_label = tk.Label(word_frame, text="", font=('Helvetica', 20), bg="lightblue", fg="darkgreen")
		self.word_label.pack(pady=10)
		
		# AI's view frame (what AI sees)
		ai_frame = tk.LabelFrame(main_frame, text="AI's View (Pattern to Solve)", font=('Helvetica', 14, 'bold'), bg="lightblue")
		ai_frame.pack(fill=tk.X, pady=10)
		
		self.ai_pattern_label = tk.Label(ai_frame, text="", font=('Helvetica', 24), bg="lightblue", fg="blue")
		self.ai_pattern_label.pack(pady=10)
		
		# Game status
		self.attempts_label = tk.Label(main_frame, text="Attempts Left: 9", font=('Helvetica', 16), bg="lightblue")
		self.attempts_label.pack(pady=5)
		
		# AI thinking display
		self.ai_thought_label = tk.Label(main_frame, text="", font=('Helvetica', 14), bg="lightblue", fg="purple")
		self.ai_thought_label.pack(pady=5)
		
		# Result display
		self.result_label = tk.Label(main_frame, text="", font=('Helvetica', 16), bg="lightblue")
		self.result_label.pack(pady=10)
		
		# Control buttons
		button_frame = tk.Frame(main_frame, bg="lightblue")
		button_frame.pack(fill=tk.X, pady=20)
		
		self.start_ai_button = tk.Button(button_frame, text="Start AI Playing", font=('Helvetica', 16), command=self.start_ai_playing, bg="green")
		self.start_ai_button.pack(side=tk.LEFT, padx=10)
		
		self.new_game_button = tk.Button(button_frame, text="New Word", font=('Helvetica', 16), command=self.start_new_game, bg="lightgreen")
		self.new_game_button.pack(side=tk.LEFT, padx=10)
		
		# Statistics display
		self.stats_text = tk.Text(main_frame, height=8, width=60, font=('Courier', 10))
		self.stats_text.pack(fill=tk.BOTH, expand=True, pady=20)
		self.update_stats_display()

	def start_new_game(self):
		# Replenish word list if empty
		if not self.words:
			self.words = words.copy()
		
		# Select and remove a word
		if self.words:
			self.selected_word = random.choice(self.words)
			self.words.remove(self.selected_word)
		else:
			self.result_label.config(text="Error: No words available!", fg='red')
			return
		
		# Reset game state
		self.game_state = GameState(self.selected_word)
		self.game_over = False
		self.ai_playing = False
		self.ai.reset()
		
		# Update UI
		self.word_label.config(text=f"Word: {self.selected_word.upper()}")
		self.ai_pattern_label.config(text=self.game_state.get_pattern())
		self.attempts_label.config(text=f"Attempts Left: {self.game_state.attempts_left}")
		self.result_label.config(text="Ready to start AI playing!", fg='black')
		self.ai_thought_label.config(text="")
		
		self.update_stats_display()

	def start_ai_playing(self):
		"""Start AI playing the game"""
		if self.game_over:
			self.start_new_game()
		
		self.ai_playing = True
		self.start_ai_button.config(state='disabled')
		
		self.result_label.config(text="AI is playing...", fg='blue')
		
		# Start AI playing in a separate thread
		threading.Thread(target=self.ai_play_loop, daemon=True).start()

	def ai_play_loop(self):
		"""AI plays the game automatically"""
		moves = []
		
		while not self.game_state.is_game_over() and self.ai_playing:
			# Get AI guess
			guess = self.ai.get_guess(self.game_state)
			moves.append(guess)
			
			# Update UI on main thread
			self.window.after(0, lambda g=guess: self.ai_thought_label.config(
				text=f"AI is thinking... Guessing: {g.upper()}"
			))
			
			# Make the guess
			is_correct, reward = self.game_state.make_guess(guess)
			
			# Update UI
			self.window.after(0, self.update_display)
			
			# Small delay for visualization
			time.sleep(1.5)
		
		# Game over
		self.window.after(0, lambda: self.finish_ai_game(moves))

	def update_display(self):
		"""Update the game display"""
		if not self.game_state:
			return
		
		# Update AI's pattern view
		pattern = self.game_state.get_pattern()
		self.ai_pattern_label.config(text=pattern)
		
		# Update attempts
		self.attempts_label.config(text=f'Attempts Left: {self.game_state.attempts_left}')
		
		# Check for game over
		if self.game_state.is_game_over():
			self.game_over = True
			self.ai_playing = False

	def finish_ai_game(self, moves):
		"""Finish AI game and update stats"""
		self.start_ai_button.config(state='normal')
		
		if self.game_state.won:
			self.result_label.config(text=f"AI WON in {len(moves)} moves!", fg='green')
			self.ai_thought_label.config(text=f"AI successfully guessed: {self.selected_word.upper()}")
			self.update_ai_stats(True)
			
			# Learn from successful game
			correct_letters = list(self.game_state.correct_letters)
			self.ai.learn_from_game(self.game_state.get_pattern().replace(' ', ''), correct_letters, True)
		else:
			self.result_label.config(text=f"AI LOST after {len(moves)} moves!", fg='red')
			self.ai_thought_label.config(text=f"AI failed to guess: {self.selected_word.upper()}")
			self.update_ai_stats(False)

	def update_ai_stats(self, won: bool):
		"""Update AI statistics and learning curve"""
		if won:
			self.ai_stats["wins"] += 1
		else:
			self.ai_stats["losses"] += 1
		
		self.ai_stats["total_games"] += 1
		self.ai_stats["learning_curve"].append(won)
		
		# Keep only last 50 games for learning curve
		if len(self.ai_stats["learning_curve"]) > 50:
			self.ai_stats["learning_curve"] = self.ai_stats["learning_curve"][-50:]
		
		self.update_stats_display()

	def update_stats_display(self):
		"""Update the statistics display"""
		stats_text = "Neural Network AI Performance:\n" + "="*50 + "\n"
		
		if self.ai_stats["total_games"] > 0:
			win_rate = (self.ai_stats["wins"] / self.ai_stats["total_games"]) * 100
			stats_text += f"Total Games: {self.ai_stats['total_games']}\n"
			stats_text += f"Wins: {self.ai_stats['wins']}\n"
			stats_text += f"Losses: {self.ai_stats['losses']}\n"
			stats_text += f"Win Rate: {win_rate:.1f}%\n\n"
			
			# Show learning curve for recent games
			if len(self.ai_stats["learning_curve"]) > 0:
				recent_games = min(10, len(self.ai_stats["learning_curve"]))
				recent_wins = sum(self.ai_stats["learning_curve"][-recent_games:])
				recent_rate = (recent_wins / recent_games) * 100
				stats_text += f"Recent Win Rate (last {recent_games}): {recent_rate:.1f}%\n"
				
				# Show learning curve as text
				curve_text = ""
				for i, won in enumerate(self.ai_stats["learning_curve"][-20:]):  # Last 20 games
					curve_text += "W" if won else "L"
					if (i + 1) % 10 == 0:
						curve_text += "\n     "
				stats_text += f"Learning Curve: {curve_text}\n"
			
			# Indicate whether a trained model is loaded
			if self.ai.model is not None:
				stats_text += "\nModel: Loaded letter_predictor.pt\n"
			else:
				stats_text += "\nModel: Not loaded (using heuristics)\n"
		else:
			stats_text += "No games played yet.\n"
		
		self.stats_text.delete(1.0, tk.END)
		self.stats_text.insert(1.0, stats_text)

# Set up the window
if __name__ == "__main__":
	window = tk.Tk()
	game = NeuralNetworkHangmanGame(window)
	window.mainloop()
