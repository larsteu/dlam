import csv
from collections import Counter


def analyze_betting_accuracy(csv_file):
    correct_predictions = 0
    total_matches = 0
    outcome_distribution = Counter()

    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            total_matches += 1

            # Extract odds and convert to float
            odds = {
                'team1': float(row['Odd team1']),
                'draw': float(row['Odd draw']),
                'team2': float(row['Odd team2'])
            }

            # Determine the predicted outcome (lowest odds)
            predicted_outcome = min(odds, key=odds.get)

            # Determine the actual outcome
            if row['Overtime/Penalty'] == 'PEN' or (row['Winner'] == 'Draw' and row['Overtime/Penalty'] != 'ET'):
                actual_outcome = 'draw'
            elif row['Winner'] == row['Team1']:
                actual_outcome = 'team1'
            elif row['Winner'] == row['Team2']:
                actual_outcome = 'team2'
            else:
                actual_outcome = 'draw'

            # Check if prediction was correct
            if predicted_outcome == actual_outcome:
                correct_predictions += 1
                print(f"{total_matches}: {row['Team1']} vs {row['Team2']} - CORRECT")
            else:
                print(f"{total_matches}: {row['Team1']} vs {row['Team2']} - INCORRECT")

            # Update outcome distribution
            outcome_distribution[actual_outcome] += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_matches if total_matches > 0 else 0

    return accuracy, outcome_distribution, total_matches


def main():
    csv_file = 'odds_em24.csv'
    accuracy, outcome_distribution, total_matches = analyze_betting_accuracy(csv_file)

    print(f"Total matches analyzed: {total_matches}")
    print(f"Betting accuracy: {accuracy:.2%}")
    print("\nOutcome distribution:")
    for outcome, count in outcome_distribution.items():
        print(f"{outcome}: {count} ({count / total_matches:.2%})")


if __name__ == "__main__":
    main()