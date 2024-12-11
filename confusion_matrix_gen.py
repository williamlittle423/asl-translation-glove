import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # File path
    csv_file = 'evaluation_results_will_on_will.csv'
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{csv_file}' is empty.")
        return
    except pd.errors.ParserError:
        print(f"Error: The file '{csv_file}' does not appear to be in CSV format.")
        return

    # Ensure the necessary columns exist
    if 'True Label' not in df.columns or 'Predicted Label' not in df.columns:
        print("Error: The CSV file must contain 'True Label' and 'Predicted Label' columns.")
        return

    # Extract true and predicted labels
    true_labels = df['True Label']
    predicted_labels = df['Predicted Label']

    # Get the sorted list of unique labels
    labels = sorted(list(set(true_labels.unique()).union(set(predicted_labels.unique()))))
    
    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    
    # Calculate total accuracy
    accuracy = accuracy_score(true_labels, predicted_labels) * 100  # Convert to percentage
    
    # Normalize the confusion matrix to 0-100 per true label (row-wise)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0 (if any row has sum 0)
    cm_normalized = np.round(cm_normalized).astype(int)  # Convert to integer percentages

    # Create a DataFrame for better visualization
    cm_df = pd.DataFrame(cm_normalized, index=labels, columns=labels)
    
    # Display the normalized confusion matrix
    print("### Motion Tracking Glove - Normalized Confusion Matrix (0-100 scale)")
    print(cm_df)
    print("\n### Total Accuracy")
    print(f"{int(round(accuracy))}%")
    
    # Optional: Save the normalized confusion matrix to a CSV file
    cm_df.to_csv('normalized_confusion_matrix.csv')
    print("\nNormalized confusion matrix has been saved to 'normalized_confusion_matrix.csv'.")
    
    # Optional: Visualize the confusion matrix using seaborn heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title('Motion Tracking Glove - Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('normalized_confusion_matrix.png')  # Save the figure
    plt.show()
    print("Confusion matrix visualization has been saved as 'normalized_confusion_matrix.png'.")

if __name__ == "__main__":
    main()
