#!/usr/bin/env python3
import pandas as pd
import sys
import os

def merge_asl_datasets(existing_file, new_file, output_file):
    """
    Merges two ASL CSV datasets into a single CSV file, ensuring consistent label columns.

    Parameters:
    - existing_file: Path to the existing dataset (e.g., 'asl_data_eric_A-L.csv')
    - new_file: Path to the new dataset to append (e.g., 'asl_data_eric_milk_want.csv')
    - output_file: Path for the merged output dataset (e.g., 'asl_data_eric.csv')
    """
    # Check if input files exist
    for file in [existing_file, new_file]:
        if not os.path.isfile(file):
            print(f"Error: The file '{file}' does not exist.")
            sys.exit(1)
    
    try:
        # Read the existing dataset
        print(f"Reading existing dataset from '{existing_file}'...")
        df_existing = pd.read_csv(existing_file)
        print(f"Existing dataset contains {len(df_existing)} records.")
    
        # Read the new dataset
        print(f"Reading new dataset from '{new_file}'...")
        df_new = pd.read_csv(new_file)
        print(f"New dataset contains {len(df_new)} records.")
    
        # Standardize label column name to 'Label' in both datasets
        if 'Letter' in df_existing.columns:
            df_existing.rename(columns={'Letter': 'Label'}, inplace=True)
            print("Renamed 'Letter' column to 'Label' in existing dataset.")
        elif 'Label' in df_existing.columns:
            print("'Label' column already exists in existing dataset.")
        else:
            print("Error: Existing dataset must have either 'Letter' or 'Label' column.")
            sys.exit(1)
        
        if 'Label' in df_new.columns:
            print("'Label' column already exists in new dataset.")
        elif 'Letter' in df_new.columns:
            df_new.rename(columns={'Letter': 'Label'}, inplace=True)
            print("Renamed 'Letter' column to 'Label' in new dataset.")
        else:
            print("Error: New dataset must have either 'Letter' or 'Label' column.")
            sys.exit(1)
    
        # Verify that both DataFrames have 'Label' column
        if 'Label' not in df_existing.columns or 'Label' not in df_new.columns:
            print("Error: One of the datasets does not have a 'Label' column after renaming.")
            sys.exit(1)
    
        # Select common columns including 'Label'
        common_columns = df_existing.columns.intersection(df_new.columns).tolist()
        print(f"Common columns to be merged: {common_columns}")
    
        # Ensure 'Label' is included
        if 'Label' not in common_columns:
            print("Error: 'Label' column is not common between datasets.")
            sys.exit(1)
    
        # Merge datasets on common columns
        df_combined = pd.concat([df_existing[common_columns], df_new[common_columns]], ignore_index=True)
        print(f"Combined dataset contains {len(df_combined)} records.")
    
        # Save the combined DataFrame to the output CSV
        print(f"Saving the combined dataset to '{output_file}'...")
        df_combined.to_csv(output_file, index=False)
        print("Merge completed successfully.")
    
    except pd.errors.EmptyDataError as e:
        print(f"Error: One of the input files is empty. {e}")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse one of the input files. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    """
    Main function to execute the merge process.
    """
    # Define input and output file paths
    existing_file = 'asl_data_eric_A-L.csv'
    new_file = 'asl_data_eric_milk_want.csv'
    output_file = 'asl_data_eric.csv'

    # Optional: Allow users to specify file paths via command-line arguments
    if len(sys.argv) > 1:
        if len(sys.argv) != 4:
            print("Usage: python3 merge_asl_data.py <existing_file> <new_file> <output_file>")
            print("Example: python3 merge_asl_data.py asl_data_eric_A-L.csv asl_data_eric_milk_want.csv asl_data_eric.csv")
            sys.exit(1)
        existing_file, new_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

    # Call the merge function
    merge_asl_datasets(existing_file, new_file, output_file)

if __name__ == "__main__":
    main()
