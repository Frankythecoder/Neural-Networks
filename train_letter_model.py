import os
import random
import json
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Data utilities
# -----------------------------

def load_words(path: str) -> List[str]:
	try:
		with open(path, 'r', encoding='utf-8') as f:
			return [w.strip().lower() for w in f.read().split(',') if w.strip()]
	except FileNotFoundError:
		return ['python','programming','computer','hangman','game','artificial','intelligence','machine','learning']

class HangmanPatternDataset(Dataset):
	"""Creates (pattern, target_letter) pairs from words by masking one or more letters."""
	def __init__(self, words: List[str], samples_per_word: int = 5, max_len: int = 20):
		self.alphabet = [chr(i) for i in range(ord('a'), ord('z')+1)]
		self.char_to_idx = {c:i for i,c in enumerate(self.alphabet)}
		self.max_len = max_len
		self.samples: List[Tuple[str, int]] = []
		for w in words:
			w = ''.join([c for c in w if c.isalpha()])
			if not w:
				continue
			for _ in range(samples_per_word):
				# choose a letter position to hide as the target
				pos_candidates = [i for i,ch in enumerate(w) if ch.isalpha()]
				if not pos_candidates:
					continue
				pos = random.choice(pos_candidates)
				target_letter = w[pos]
				pattern_chars = [c if (i!=pos and random.random()<0.6) else '_' for i,c in enumerate(w)]
				pattern = ''.join(pattern_chars)
				self.samples.append((pattern, self.char_to_idx[target_letter]))

	def __len__(self):
		return len(self.samples)

	def encode_pattern(self, pattern: str) -> torch.Tensor:
		# Encode as one-hot over 27 symbols per position (26 letters + underscore)
		symbols = ['_'] + self.alphabet
		sym_to_idx = {s:i for i,s in enumerate(symbols)}
		max_len = self.max_len
		encoded = torch.zeros(max_len, len(symbols), dtype=torch.float32)
		p = pattern[:max_len].ljust(max_len, '_')
		for i,ch in enumerate(p):
			encoded[i, sym_to_idx.get(ch, 0)] = 1.0
		return encoded.view(-1)  # flatten

	def __getitem__(self, idx):
		pattern, target_idx = self.samples[idx]
		x = self.encode_pattern(pattern)
		y = torch.tensor(target_idx, dtype=torch.long)
		return x, y

# -----------------------------
# Model
# -----------------------------

class LetterPredictor(nn.Module):
	def __init__(self, input_size: int, hidden_size: int = 512, num_classes: int = 26):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, num_classes),
		)

	def forward(self, x):
		return self.net(x)

# -----------------------------
# Training loop
# -----------------------------

def train_model(word_path: str = 'Hangman_wordbank', out_dir: str = 'models', epochs: int = 10, batch_size: int = 128, lr: float = 1e-3, samples_per_word: int = 8, max_len: int = 20, seed: int = 42):
	random.seed(seed)
	torch.manual_seed(seed)
	os.makedirs(out_dir, exist_ok=True)

	words = load_words(word_path)
	random.shuffle(words)
	split = int(0.9 * len(words))
	train_words, val_words = words[:split], words[split:]

	train_ds = HangmanPatternDataset(train_words, samples_per_word=samples_per_word, max_len=max_len)
	val_ds = HangmanPatternDataset(val_words, samples_per_word=samples_per_word, max_len=max_len)

	input_size = (max_len) * (26 + 1)
	model = LetterPredictor(input_size)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = nn.CrossEntropyLoss()

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

	best_val_acc = 0.0
	for epoch in range(1, epochs+1):
		model.train()
		total, correct, loss_sum = 0, 0, 0.0
		for x, y in train_loader:
			x = x.to(device)
			y = y.to(device)
			optimizer.zero_grad()
			logits = model(x)
			loss = criterion(logits, y)
			loss.backward()
			optimizer.step()
			loss_sum += float(loss.item())
			pred = logits.argmax(dim=1)
			correct += int((pred == y).sum().item())
			total += y.size(0)
		train_acc = correct / max(1, total)

		model.eval()
		vtot, vcorr = 0, 0
		with torch.no_grad():
			for x, y in val_loader:
				x = x.to(device)
				y = y.to(device)
				logits = model(x)
				pred = logits.argmax(dim=1)
				vcorr += int((pred == y).sum().item())
				vtot += y.size(0)
		val_acc = vcorr / max(1, vtot)

		print(f"Epoch {epoch:02d} | TrainLoss {loss_sum/len(train_loader):.4f} | TrainAcc {train_acc*100:.1f}% | ValAcc {val_acc*100:.1f}%")

		if val_acc >= best_val_acc:
			best_val_acc = val_acc
			save_path = os.path.join(out_dir, 'letter_predictor.pt')
			torch.save({
				'state_dict': model.state_dict(),
				'config': {
					'input_size': input_size,
					'hidden_size': 512,
					'max_len': max_len,
				}
			}, save_path)
			print(f"Saved best model to {save_path} (ValAcc {val_acc*100:.1f}%)")

if __name__ == '__main__':
	train_model()
