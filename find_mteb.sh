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

# Define task lists
declare -A TASK_LISTS
TASK_LISTS[CLASSIFICATION]="AmazonCounterfactualClassification AmazonPolarityClassification AmazonReviewsClassification Banking77Classification EmotionClassification ImdbClassification MassiveIntentClassification MassiveScenarioClassification MTOPDomainClassification MTOPIntentClassification ToxicConversationsClassification TweetSentimentExtractionClassification"
TASK_LISTS[CLUSTERING]="ArxivClusteringP2P ArxivClusteringS2S BiorxivClusteringP2P BiorxivClusteringS2S MedrxivClusteringP2P MedrxivClusteringS2S RedditClustering RedditClusteringP2P StackExchangeClustering StackExchangeClusteringP2P TwentyNewsgroupsClustering"
TASK_LISTS[PAIR_CLASSIFICATION]="SprintDuplicateQuestions TwitterSemEval2015 TwitterURLCorpus"
TASK_LISTS[RERANKING]="AskUbuntuDupQuestions MindSmallReranking SciDocsRR StackOverflowDupQuestions"
TASK_LISTS[RETRIEVAL]="ArguAna ClimateFEVER DBPedia FEVER FiQA2018 HotpotQA MSMARCO NFCorpus NQ QuoraRetrieval SCIDOCS SciFact Touche2020 TRECCOVID"
TASK_LISTS[STS]="BIOSSES SICK-R STS12 STS13 STS14 STS15 STS16 STS17 STS22 STSBenchmark SummEval"
TASK_LISTS[CQA_DUPSTACK]="CQADupstackAndroidRetrieval CQADupstackEnglishRetrieval CQADupstackGamingRetrieval CQADupstackGisRetrieval CQADupstackMathematicaRetrieval CQADupstackPhysicsRetrieval CQADupstackProgrammersRetrieval CQADupstackStatsRetrieval CQADupstackTexRetrieval CQADupstackUnixRetrieval CQADupstackWebmastersRetrieval CQADupstackWordpressRetrieval"

# Initialize arrays for each task category
for category in "${!TASK_LISTS[@]}"; do
    declare -a "${category}_scores"
    declare -a "${category}_files"
done

# Initialize CQADupstack-specific variables
CQA_DUPSTACK_count=0

# Create CSV file
csv_file="results.csv"
echo "Filename,Score,Category" > "$csv_file"

# The main command
while IFS= read -r line; do
    filename=$(echo "$line" | cut -d' ' -f1)
    score=$(echo "$line" | cut -d' ' -f2)
    subset=$(echo "$line" | cut -d' ' -f3)
    
    echo "$filename $score $subset"
    
    # Remove .json extension for matching
    task_name=${filename%.json}
    
    for category in "${!TASK_LISTS[@]}"; do
        if [[ "${TASK_LISTS[$category]}" == *"$task_name"* ]]; then
            eval "${category}_scores+=(${score})"
            eval "${category}_files+=(\"${filename}\")"
            # Count CQADupstack files
            if [ "$category" == "CQA_DUPSTACK" ]; then
                ((CQA_DUPSTACK_count++))
            fi
            # Add to CSV file
            echo "$filename,$score,$category" >> "$csv_file"
            break
        fi
    done
done < <(find "$search_dir" -type f -name "*.json" -print0 | xargs -0 -I {} sh -c '
jq -r ".. | objects | select((.hf_subset? == \"en\" or .hf_subset? == \"default\") and .main_score?) | \"\(.main_score) \(.hf_subset)\"" {} | 
while read score subset; do 
    formatted_score=$(printf "%.2f" $(echo "$score * 100" | bc))
    echo "$(basename {}) $formatted_score $subset"
done
')

# Function to calculate average
calculate_average() {
    local scores=("${!1}")
    if [ ${#scores[@]} -eq 0 ]; then
        echo "0"
    else
        sum=$(echo "${scores[@]}" | tr ' ' '+' | bc)
        echo "scale=2; $sum / ${#scores[@]}" | bc
    fi
}

# Function to print results
print_results() {
    local scores=("${!1}")
    local files=("${!2}")
    local name=$3
    
    if [ ${#scores[@]} -ne 0 ]; then
        average=$(calculate_average "$1")
        echo ""
        echo "Average of $name scores: $average"
        echo "$name files included:"
        printf '%s\n' "${files[@]}" | sort | uniq
    else
        echo ""
        echo "No files found for $name"
    fi
}

# Calculate and print averages for each category
for category in "${!TASK_LISTS[@]}"; do
    if [ "$category" != "RETRIEVAL" ] && [ "$category" != "CQA_DUPSTACK" ]; then
        scores_var="${category}_scores[@]"
        files_var="${category}_files[@]"
        print_results "$scores_var" "$files_var" "$category"
    fi
done

# Handle CQA_DUPSTACK category
if [ $CQA_DUPSTACK_count -ne 0 ]; then
    cqa_sum=$(echo "${CQA_DUPSTACK_scores[@]}" | tr ' ' '+' | bc)
    cqa_average=$(echo "scale=2; $cqa_sum / $CQA_DUPSTACK_count" | bc)
else
    cqa_average=0
fi

echo ""
echo "Average of CQA_DUPSTACK scores: $cqa_average"
echo "CQA_DUPSTACK files included:"
printf '%s\n' "${CQA_DUPSTACK_files[@]}" | sort | uniq
echo "Number of CQA_DUPSTACK files: $CQA_DUPSTACK_count"

# Handle RETRIEVAL category
retrieval_scores=(${RETRIEVAL_scores[@]} $cqa_average)
retrieval_count=$((${#RETRIEVAL_scores[@]} + 1))  # +1 for CQADupstack average
retrieval_sum=$(echo "${retrieval_scores[@]}" | tr ' ' '+' | bc)
retrieval_average=$(echo "scale=2; $retrieval_sum / $retrieval_count" | bc)

echo ""
echo "Average of RETRIEVAL scores (including CQA_DUPSTACK average): $retrieval_average"
echo "RETRIEVAL files included:"
printf '%s\n' "${RETRIEVAL_files[@]}" | sort | uniq
echo "Number of RETRIEVAL scores (including CQA_DUPSTACK as one score): $retrieval_count"

echo ""
echo "Results have been saved to $csv_file"
