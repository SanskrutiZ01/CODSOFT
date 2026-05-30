# ==========================================
# CodSoft AI Internship - Task 2
# Tic-Tac-Toe AI using Minimax
# Author: Sanskruti Zode
# ==========================================

def play_game():

    board = [" " for _ in range(9)]

    def show_board():
        print()
        print(f" {board[0]} | {board[1]} | {board[2]}")
        print("---+---+---")
        print(f" {board[3]} | {board[4]} | {board[5]}")
        print("---+---+---")
        print(f" {board[6]} | {board[7]} | {board[8]}")
        print()

    def check_winner(player):
        wins = [
            [0,1,2],
            [3,4,5],
            [6,7,8],
            [0,3,6],
            [1,4,7],
            [2,5,8],
            [0,4,8],
            [2,4,6]
        ]

        for combo in wins:
            if all(board[i] == player for i in combo):
                return True
        return False

    def check_draw():
        return " " not in board

    def minimax(is_maximizing):

        if check_winner("O"):
            return 1

        if check_winner("X"):
            return -1

        if check_draw():
            return 0

        if is_maximizing:
            best_score = -100

            for i in range(9):
                if board[i] == " ":
                    board[i] = "O"
                    score = minimax(False)
                    board[i] = " "
                    best_score = max(score, best_score)

            return best_score

        else:
            best_score = 100

            for i in range(9):
                if board[i] == " ":
                    board[i] = "X"
                    score = minimax(True)
                    board[i] = " "
                    best_score = min(score, best_score)

            return best_score

    def ai_move():
        best_score = -100
        move = -1

        for i in range(9):
            if board[i] == " ":
                board[i] = "O"

                score = minimax(False)

                board[i] = " "

                if score > best_score:
                    best_score = score
                    move = i

        board[move] = "O"

    def player_move():
        while True:
            try:
                move = int(input("Enter position (1-9): ")) - 1

                if 0 <= move <= 8 and board[move] == " ":
                    board[move] = "X"
                    break
                else:
                    print("Invalid move. Try again.")

            except ValueError:
                print("Please enter a number from 1 to 9.")

    print("\n🎮 Tic-Tac-Toe AI using Minimax")
    print("You = X | Computer = O")
    print("Positions:")
    print("1 | 2 | 3")
    print("4 | 5 | 6")
    print("7 | 8 | 9")

    while True:

        show_board()

        player_move()

        if check_winner("X"):
            show_board()
            print("🎉 Congratulations! You win!")
            break

        if check_draw():
            show_board()
            print("🤝 It's a draw!")
            break

        print("Computer is thinking...")
        ai_move()

        if check_winner("O"):
            show_board()
            print("🤖 Computer wins!")
            break

        if check_draw():
            show_board()
            print("🤝 It's a draw!")
            break


# Main program
while True:
    play_game()

    again = input("Do you want to play again? (yes/no): ").lower().strip()

    if again != "yes":
        print("👋 Thanks for playing!")
        break