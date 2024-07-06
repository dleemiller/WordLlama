#!/bin/bash

# Check if a directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# The directory to search in
search_dir="$1"

# Check if the provided argument is a directory
if [ ! -d "$search_dir" ]; then
    echo "Error: '$search_dir' is not a directory"
    exit 1
fi

# Arrays to store scores and filenames
declare -a classification_scores
declare -a classification_files
declare -a clustering_scores
declare -a clustering_files

# The main command
while IFS= read -r line; do
    filename=$(echo "$line" | cut -d' ' -f1)
    score=$(echo "$line" | cut -d' ' -f2)
    subset=$(echo "$line" | cut -d' ' -f3)
    
    echo "$filename $score $subset"
    
    if [[ $filename == *Classification.json ]]; then
        classification_scores+=($score)
        classification_files+=("$filename")
    elif [[ $filename == *Clustering*.json ]]; then
        clustering_scores+=($score)
        clustering_files+=("$filename")
    fi
done < <(find "$search_dir" -type f -name "*.json" -print0 | xargs -0 -I {} sh -c '
jq -r ".. | objects | select((.hf_subset? == \"en\" or .hf_subset? == \"default\") and .main_score?) | \"\(.main_score) \(.hf_subset)\"" {} | 
while read score subset; do 
    formatted_score=$(printf "%.2f" $(echo "$score * 100" | bc))
    echo "$(basename {}) $formatted_score $subset"
done
')

# Function to calculate and print average
calculate_average() {
    local -n scores=$1
    local -n files=$2
    local name=$3
    
    if [ ${#scores[@]} -ne 0 ]; then
        sum=0
        for score in "${scores[@]}"; do
            sum=$(echo "$sum + $score" | bc)
        done
        average=$(echo "scale=2; $sum / ${#scores[@]}" | bc)
        echo ""
        echo "Average of $name scores: $average"
        echo "$name files included:"
        printf '%s\n' "${files[@]}" | sort | uniq
    fi
}

# Calculate and print averages
calculate_average classification_scores classification_files "Classification"
calculate_average clustering_scores clustering_files "Clustering"
