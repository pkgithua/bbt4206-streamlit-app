[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ccGdno4Y)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=21458843)
# Sentiment Analysis

| Key              | Value                                                                                                                                                                                                                                                                                              |
|:-----------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Course Codes** | BBT 4206, BFS 4102                                                                                                                                                                                                                                                                                 |
| **Course Names** | BBT 4206: Business Intelligence II (Week 10-12 of 13) and <br/>BFS 4102: Advanced Business Data Analytics (Week 10-12 of 13)                                                                                                                                                                       |
| **Semester**     | August to November 2025                                                                                                                                                                                                                                                                            |
| **Lecturer**     | Allan Omondi                                                                                                                                                                                                                                                                                       |
| **Contact**      | aomondi@strathmore.edu                                                                                                                                                                                                                                                                             |
| **Note**         | The lecture contains both theory and practice.<br/>This notebook forms part of the practice.<br/>It is intended for educational purpose only.<br/>Recommended citation: [BibTex](https://raw.githubusercontent.com/course-files/NaturalLanguageProcessing/refs/heads/main/RecommendedCitation.bib) |

## Repository Structure

```text
.
├── 1_topic_modeling_using_LDA.ipynb
├── 2_sentiment_analysis.ipynb
├── analysis
│   ├── absenteeism_vs_rating.png
│   ├── absenteeism_vs_timing.png
│   ├── analysis_output.txt
│   ├── analysis_report.txt
│   └── nlp_analysis.py
├── data
│   ├── course_evaluation.csv
│   ├── processed_scaled_down_reviews.csv
│   ├── processed_scaled_down_reviews_with_topics.csv
│   └── processed_scaled_down_reviews_with_topics_and_sentiments.csv
├── model
│   ├── sentiment_classifier.pkl
│   ├── topic_labels.json
│   ├── topic_model_lda.pkl
│   ├── topic_vectorizer.pkl
│   └── topic_vectorizer_using_tfidf.pkl
├── README.md
├── LICENSE
├── lab_submission_instructions.md
├── requirements.txt
├── setup_instructions.md
└── RecommendedCitation.bib
```

## Setup Instructions

1. Install **Python 3.11** (wheels for `wordcloud` are only available up to 3.11).
2. Create and activate the virtual environment:
   ```powershell
   py -3.11 -m venv .venv311
   .\.venv311\Scripts\Activate
   ```
3. Install the dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. (Optional) Update PyCharm/VS Code to use `.venv311`.

For the official lecturer instructions, refer to [setup_instructions.md](setup_instructions.md).

## Lab Manual

- Topic Modelling walkthrough: [1_topic_modeling_using_LDA.ipynb](1_topic_modeling_using_LDA.ipynb)  
- Sentiment analysis walkthrough: [2_sentiment_analysis.ipynb](2_sentiment_analysis.ipynb)

## New Additions for the Course-Evaluation Study

### Dataset
- `data/course_evaluation.csv` (130 anonymised responses from BI I & II).  
  Source: [course-files/NaturalLanguageProcessing](https://raw.githubusercontent.com/course-files/NaturalLanguageProcessing/refs/heads/main/data/202511-ft_bi1_bi2_course_evaluation.csv)

### Scripted Analysis
- `analysis/nlp_analysis.py` orchestrates the qualitative study.  
- Run it from the repo root (with `.venv311` activated):
  ```powershell
  python .\analysis\nlp_analysis.py
  ```
- Outputs:
  - `analysis/analysis_output.txt`: full console log
  - `analysis/analysis_report.txt`: concise summary
  - `analysis/absenteeism_vs_rating.png`
  - `analysis/absenteeism_vs_timing.png`

### Insights Covered
- Topic differences across class groups (A/B/C)
- Contrast of low vs high “I enjoyed the course” ratings
- Correlations between absenteeism, overall rating, and punctuality perception
- Deep dive on “more engagement” recommendations (themes + expectations)

Use these artefacts when writing up the recommendation section or populating the submission template in `lab_submission_instructions.md`.

## Lab Submission Instructions

See [lab_submission_instructions.md](lab_submission_instructions.md) for the formal template (team info, video link, hosted UI link, etc.). This repo already contains:
- Topic & sentiment modelling notebooks
- Trained artefacts under `model/`
- The qualitative analysis report + visuals under `analysis/`

Update those sections with your team’s details, interpretation, recommendations, video demo, and hosted interface link before the final submission.
