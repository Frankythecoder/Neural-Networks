# Hangman Game

# Neural Network AI Hangman

## Description
A specialized Hangman game featuring a Neural Network AI that learns and plays autonomously. Watch as the AI attempts to guess words using advanced pattern recognition and machine learning techniques. The word is visible to you (the observer) while the AI works with only the pattern of blanks.

## Features

### 🤖 Neural Network AI Player
- **Advanced Pattern Recognition**: Analyzes word structures and letter positions
- **Learning Capability**: Improves performance through successful pattern recognition
- **Context-Aware Guessing**: Considers word length, position, and revealed letters
- **Adaptive Strategy**: Learns from each game and builds a knowledge base
- **Optional Trained Model**: Uses a PyTorch model if available

## Requirements
- Python 3.8+
- tkinter (built into Python)
- numpy >= 1.21.0
- PyTorch CPU (optional for offline-trained model)

Install dependencies:
```bash
pip install -r requirements.txt
# On Windows CPU, if torch fails via requirements:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Run the Game
```bash
python neural_network_hangman.py
```
- If a trained model is found at `models/letter_predictor.pt`, the AI will use it (display shows "Model: Loaded").
- Otherwise, it falls back to heuristic pattern logic (display shows "Model: Not loaded").

## Train the Neural Network (Epoch-based)
We provide a simple PyTorch training script that learns to predict the next best letter from masked word patterns.

Train with defaults (10 epochs):
```bash
python train_letter_model.py
```
Key details:
- Input: patterns generated from `Hangman_wordbank` by masking random positions
- Model: 2-layer MLP (512 hidden units)
- Saves best checkpoint to `models/letter_predictor.pt`
- Config is stored inside the checkpoint for seamless loading

Adjust training options by editing `train_letter_model.py`:
- `epochs`, `batch_size`, `lr`, `samples_per_word`, `max_len`

After training, re-run the game and the UI will show that the model is loaded.

## How It Works
- The UI shows the full word to you, but the AI sees only the masked pattern.
- On each step, the AI chooses a letter:
  - If a trained model is loaded: ranks letters via softmax probabilities
  - Otherwise: uses heuristic pattern/position analysis and learned successful patterns

## Files
```
neural_network_hangman.py   # Game with optional PyTorch inference
train_letter_model.py       # Epoch-based training script (PyTorch)
models/letter_predictor.pt  # Saved model (created after training)
Hangman_wordbank            # Word list (comma-separated)
requirements.txt            # Dependencies (see torch note for Windows)
README_AI.md                # This documentation
```

## Tips
- More diverse words in `Hangman_wordbank` → better generalization
- Increase `samples_per_word` and `epochs` for stronger models
- CPU training is fine for this scale; GPU is optional

## Troubleshooting
- Torch install on Windows CPU:
  - `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Model not loading:
  - Ensure `models/letter_predictor.pt` exists
  - Re-run training; check console logs for "Saved best model"
- No improvement:
  - Increase epochs, samples, or enrich the word list


## Features(Gameplay)
- Randomly selects words from a word bank.
- Allows players to guess letters one at a time.
- Displays hints when attempts reach 6 or 3.
- Tracks incorrect guesses and provides visual feedback.
- Announces win/loss conditions and allows restarting the game.
- GUI implemented using Tkinter for easy interaction.


## Installation & Setup
1. Clone or download the repository.
2. Ensure Python is installed on your system.
3. Place your word bank file at the specified path in the code or modify `file_path` accordingly.
4. Run the script using:
   ```sh
   python hangman.py
   ```

## File Structure
- `hangman.py`: Main Python script that runs the game.
- `words/Hangman_wordbank`: CSV file containing a list of words for the game.

## How to Play
1. Launch the game by running `hangman.py`.
2. A random word will be chosen and displayed as underscores.
3. Type a letter and press "Guess" to check if it's in the word.
4. You have 9 attempts to guess the word correctly.
5. If you fail, the word will be revealed.
6. Click "New Game" to start again.

## Customization
- Modify the word bank file to include your own words.
- Adjust UI settings by modifying the Tkinter components in `create_ui()`.
- Change the number of attempts by updating `self.attempts_left` in `start_new_game()`.

## Author
Created by Frankythecoder

## License
This project is open-source. Modify and distribute as needed!