#!/usr/bin/env python3
"""
Neural Network AI Hangman Demo Script
Demonstrates the Neural Network AI capabilities without GUI
"""

import random
import time
from neural_network_hangman import GameState, NeuralNetworkAI

def load_words():
    """Load words from the wordbank"""
    try:
        with open('Hangman_wordbank', 'r', encoding='utf-8') as f:
            words = [word.strip().lower() for word in f.read().split(',') if word.strip()]
    except FileNotFoundError:
        words = ['python', 'programming', 'computer', 'hangman', 'game', 'artificial', 'intelligence']
    return words

def run_ai_demo(ai, word, game_number):
    """Run a single AI game and return results"""
    game_state = GameState(word)
    ai.reset()
    moves = []
    
    print(f"\n🎯 Game {game_number}: AI playing word: {word.upper()}")
    print(f"📝 Initial pattern: {game_state.get_pattern()}")
    print("🤖 AI thinking process:")
    
    while not game_state.is_game_over():
        guess = ai.get_guess(game_state)
        moves.append(guess)
        is_correct, reward = game_state.make_guess(guess)
        
        status = "✅ Correct!" if is_correct else "❌ Wrong"
        print(f"   Guess: {guess.upper()} | Pattern: {game_state.get_pattern()} | Attempts: {game_state.attempts_left} | {status}")
        
        if game_state.is_game_over():
            break
    
    result = "🎉 WON" if game_state.won else "💀 LOST"
    print(f"🏁 Result: {result} in {len(moves)} moves")
    
    # Learn from the game
    if game_state.won:
        correct_letters = list(game_state.correct_letters)
        ai.learn_from_game(game_state.get_pattern().replace(' ', ''), correct_letters, True)
    
    return game_state.won, len(moves)

def main():
    """Run Neural Network AI demonstration"""
    print("🧠 Neural Network AI Hangman Demonstration")
    print("=" * 60)
    print("This demo shows how the AI learns and improves over time!")
    print("=" * 60)
    
    # Load words
    words = load_words()
    test_words = random.sample(words, min(10, len(words)))
    
    # Initialize AI
    ai = NeuralNetworkAI()
    
    # Track performance
    performance = {"wins": 0, "total_moves": 0, "games": 0, "learning_curve": []}
    
    print(f"🎲 Testing on {len(test_words)} words: {[w.upper() for w in test_words]}")
    print("\n" + "="*60)
    
    # Run games
    for i, word in enumerate(test_words, 1):
        won, moves = run_ai_demo(ai, word, i)
        
        # Update performance
        performance["wins"] += won
        performance["total_moves"] += moves
        performance["games"] += 1
        performance["learning_curve"].append(won)
        
        # Show current stats
        win_rate = (performance["wins"] / performance["games"]) * 100
        avg_moves = performance["total_moves"] / performance["games"]
        
        print(f"\n📊 Current Stats: {performance['wins']}/{performance['games']} wins ({win_rate:.1f}%) | Avg moves: {avg_moves:.1f}")
        
        # Show learning curve
        if len(performance["learning_curve"]) >= 5:
            recent = performance["learning_curve"][-5:]
            recent_wins = sum(recent)
            print(f"📈 Recent performance (last 5): {recent_wins}/5 wins")
        
        print("-" * 60)
        time.sleep(1)  # Pause between games for readability
    
    # Display final results
    print(f"\n{'='*60}")
    print("🏆 FINAL RESULTS")
    print(f"{'='*60}")
    
    win_rate = (performance["wins"] / performance["games"]) * 100
    avg_moves = performance["total_moves"] / performance["games"]
    
    print(f"🎯 Total Games: {performance['games']}")
    print(f"✅ Wins: {performance['wins']}")
    print(f"❌ Losses: {performance['games'] - performance['wins']}")
    print(f"📊 Win Rate: {win_rate:.1f}%")
    print(f"🎲 Average Moves: {avg_moves:.1f}")
    
    # Show learning curve
    if len(performance["learning_curve"]) > 0:
        print(f"\n📈 Learning Curve (W=Win, L=Loss):")
        curve_text = ""
        for i, won in enumerate(performance["learning_curve"]):
            curve_text += "W" if won else "L"
            if (i + 1) % 10 == 0:
                curve_text += "\n     "
        print(f"   {curve_text}")
    
    # Show learned patterns
    if hasattr(ai, 'word_patterns') and ai.word_patterns:
        print(f"\n🧠 AI Learned Patterns: {len(ai.word_patterns)}")
        print("   The AI has built a knowledge base of successful word patterns!")
    
    print(f"\n🎉 Demo complete! The AI showed learning and adaptation over {performance['games']} games.")

if __name__ == "__main__":
    main()
