# Hangman Game

## Description
This is a simple Hangman game implemented using Python and Tkinter for the graphical user interface (GUI). The game selects words from a predefined word bank stored in a CSV file and allows players to guess letters to reveal the hidden word.

## Features
- Randomly selects words from a word bank.
- Allows players to guess letters one at a time.
- Displays hints when attempts reach 6 or 3.
- Tracks incorrect guesses and provides visual feedback.
- Announces win/loss conditions and allows restarting the game.
- GUI implemented using Tkinter for easy interaction.

## Requirements
- Python 3.x
- Tkinter (built into Python standard library)

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

Agent testing 2

Agent testing 8

Agent testing 10