import tkinter as tk
import random
import csv
import os

# Function to load words from the provided wordbank
file_path = r'C:\Users\D.Frank\OneDrive\Documents\hangman\words\Hangman_wordbank'
def load_words_from_csv(file_path):
    words = []
    
    try:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            words = [word.strip().lower() for word in csvfile.read().split(',') if word.strip()]
    except FileNotFoundError:
        print(f"Word bank file not found at {file_path}")
        words = ['python', 'programming', 'computer', 'hangman', 'game']
    
    return words

# Load words from the provided wordbank
words = load_words_from_csv(file_path)

class HangmanGame:
    def __init__(self, window):
        self.window = window
        self.window.title("Hangman Game")
        self.window.config(bg="lightblue")
        self.window.geometry("800x800")

        # Game variables
        self.words = words.copy()  # Creating a copy to remove words as they're used
        self.selected_word = ""
        self.guessed_letters = []
        self.attempts_left = 9
        self.incorrect_guesses = 0
        self.game_over = False  # New flag to track game state

        # Create UI elements
        self.create_ui()

        # Start first game
        self.start_new_game()

    def create_ui(self):
        # Labels
        self.word_label = tk.Label(self.window, text="Word to guess", font=('Helvetica', 24), bg="lightblue")
        self.word_label.pack(pady=20)

        self.attempts_label = tk.Label(self.window, text="Attempts Left: 9", font=('Helvetica', 18), bg="lightblue")
        self.attempts_label.pack(pady=10)

        self.result_label = tk.Label(self.window, text="Welcome to Hangman!", font=('Helvetica', 16), bg="lightblue")
        self.result_label.pack(pady=20)

        # Entry box for user input
        self.guess_entry = tk.Entry(self.window, font=('Helvetica', 16), width=5)
        self.guess_entry.pack(pady=10)
        self.guess_entry.bind('<Return>', lambda event: self.guess_letter())

        # Guess button
        self.guess_button = tk.Button(self.window, text="Guess", font=('Helvetica', 16), command=self.guess_letter, bg="yellow")
        self.guess_button.pack(pady=10)

        # New Game button
        self.new_game_button = tk.Button(self.window, text="New Game", font=('Helvetica', 16), command=self.start_new_game, bg="green")
        self.new_game_button.pack(pady=20)

    def start_new_game(self):
        # Replenish word list if empty
        if not self.words:
            self.words = words.copy()

        # Select and remove a word to prevent repetition
        if self.words:
            self.selected_word = random.choice(self.words)
            self.words.remove(self.selected_word)
        else:
            self.result_label.config(text="Error: No words available!", fg='red')
            return

        # Reset game state
        self.guessed_letters.clear()
        self.attempts_left = 9
        self.incorrect_guesses = 0
        self.game_over = False  # Reset game over flag

        # Show input elements
        self.guess_entry.pack(pady=10)
        self.guess_button.pack(pady=10)

        # Update UI
        self.result_label.config(text="Welcome to Hangman!", fg='black')
        self.update_display()

    def get_hint(self):
        # Find indices of letters not yet guessed
        missing_indices = [i for i, letter in enumerate(self.selected_word) if letter not in self.guessed_letters]
        
        if missing_indices:
            # Choose a random unguessed letter
            hint_index = random.choice(missing_indices)
            return self.selected_word[hint_index], hint_index + 1  # Convert to 1-based index
        return None, None

    def guess_letter(self):
        # If game is over, show game over message
        if self.game_over:
            self.result_label.config(text="Game is over!", fg='red')
            return

        guessed_letter = self.guess_entry.get().lower()
        self.guess_entry.delete(0, tk.END)

        if len(guessed_letter) == 1 and guessed_letter.isalpha():
            if guessed_letter in self.guessed_letters:
                self.result_label.config(text="You already guessed that letter!", fg='orange')
            elif guessed_letter in self.selected_word:
                self.guessed_letters.append(guessed_letter)
                self.result_label.config(text="Good guess!", fg='blue')
            else:
                self.guessed_letters.append(guessed_letter)
                self.attempts_left -= 1
                self.incorrect_guesses += 1
                
                # Provide hint when attempts reach 6 or 3
                if self.attempts_left == 6 or self.attempts_left == 3:
                    hint, position = self.get_hint()
                    if hint:
                        self.result_label.config(text=f"Hint: The letter '{hint}' is in position {position}", fg='purple')
                else:
                    self.result_label.config(text="Oops! Wrong guess.", fg='red')
        else:
            self.result_label.config(text="Please enter a valid letter!", fg='purple')
        
        self.update_display()

    def update_display(self):
        # Display word with guessed letters revealed
        word_display = ' '.join([letter if letter in self.guessed_letters else '_' for letter in self.selected_word])
        
        self.word_label.config(text=word_display)
        self.attempts_label.config(text=f'Attempts Left: {self.attempts_left}')
        
        # Check for win or lose conditions
        if all(letter in self.guessed_letters for letter in self.selected_word):
            self.result_label.config(text="Congratulations! You won!", fg='green')
            self.game_over = True
            # Hide input elements when game is over
            self.guess_entry.pack_forget()
            self.guess_button.pack_forget()
        elif self.attempts_left <= 0:
            self.result_label.config(text=f"Game Over! The word was: {self.selected_word}", fg='red')
            self.game_over = True
            # Hide input elements when game is over
            self.guess_entry.pack_forget()
            self.guess_button.pack_forget()

# Set up the window
window = tk.Tk()
game = HangmanGame(window)
window.mainloop()