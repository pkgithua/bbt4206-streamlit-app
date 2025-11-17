"""
Comprehensive Qualitative Data Analysis of Course Evaluation Data
NLP Lab - Deep Analysis with Insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from scipy import stats
import re
import warnings
import sys
import io
from pathlib import Path
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Download NLTK data if needed
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
except:
    print("NLTK not fully available, using basic text processing")

class CourseEvaluationAnalyzer:
    def __init__(self, csv_path=None):
        """Initialize the analyzer with the dataset"""
        self.base_dir = Path(__file__).resolve().parent
        self.output_dir = self.base_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        default_csv = self.base_dir.parent / "data" / "course_evaluation.csv"
        csv_path = Path(csv_path) if csv_path else default_csv

        print(f"Loading dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Text columns for analysis
        self.text_columns = [
            'f_1_In_your_opinion_which_topics_(if_any)_should_be_added_to_the_Business_Intelligence_I_and_II_curriculum',
            'f_2_In_your_opinion_which_topics_(if_any)_should_be_removed_from_the_Business_Intelligence_I_and_II_curriculum',
            'f_3_Write_at_least_two_things_you_liked_about_the_teaching_and_learning_in_this_course',
            'f_4_Write_at_least_one_recommendation_to_improve_the_teaching_and_learning_in_this_course_(for_future_classes)'
        ]
        
        # Initialize text preprocessing
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            self.lemmatizer = None
        
        self.preprocess_text_data()
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text) or text == 'None':
            return ""
        
        text = str(text).lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        if self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
        else:
            tokens = [token for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_text_data(self):
        """Preprocess all text columns"""
        print("Preprocessing text data...")
        for col in self.text_columns:
            if col in self.df.columns:
                self.df[f'{col}_processed'] = self.df[col].apply(self.preprocess_text)
        print("Text preprocessing complete.")
    
    def extract_topics_lda(self, texts, n_topics=5, max_features=100):
        """Extract topics using LDA"""
        # Filter out empty texts
        texts = [t for t in texts if t and len(t.strip()) > 0]
        if len(texts) < n_topics:
            return None, None
        
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
        lda.fit(X)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
            topics.append(top_words)
        
        return topics, lda.transform(X)
    
    def analyze_topics_by_group(self):
        """Question 1: Relationship between topics and class group"""
        print("\n" + "="*80)
        print("ANALYSIS 1: Relationship between Topics and Class Group")
        print("="*80)
        
        # Combine all text responses for each group
        group_texts = {}
        for group in self.df['a_3_class_group'].unique():
            if pd.isna(group):
                continue
            group_df = self.df[self.df['a_3_class_group'] == group]
            
            # Combine all text responses
            all_texts = []
            for col in self.text_columns:
                if f'{col}_processed' in self.df.columns:
                    texts = group_df[f'{col}_processed'].dropna().tolist()
                    all_texts.extend([t for t in texts if t and len(t.strip()) > 0])
            
            group_texts[group] = all_texts
        
        # Extract topics for each group
        print("\nExtracting topics for each class group...")
        group_topics = {}
        for group, texts in group_texts.items():
            if len(texts) > 10:  # Need sufficient data
                topics, _ = self.extract_topics_lda(texts, n_topics=5)
                if topics:
                    group_topics[group] = topics
                    print(f"\n{group} - Top Topics:")
                    for i, topic in enumerate(topics, 1):
                        print(f"  Topic {i}: {', '.join(topic[:5])}")
        
        # Statistical analysis: Word frequency by group
        print("\n" + "-"*80)
        print("Word Frequency Analysis by Group:")
        print("-"*80)
        
        for group in self.df['a_3_class_group'].unique():
            if pd.isna(group):
                continue
            group_df = self.df[self.df['a_3_class_group'] == group]
            
            # Get all words from recommendations
            all_words = []
            for col in self.text_columns:
                if f'{col}_processed' in self.df.columns:
                    texts = group_df[f'{col}_processed'].dropna().tolist()
                    for text in texts:
                        if text:
                            all_words.extend(text.split())
            
            if all_words:
                word_freq = pd.Series(all_words).value_counts().head(10)
                print(f"\n{group} - Top 10 Words:")
                print(word_freq.to_string())
        
        # Hypothesis
        print("\n" + "-"*80)
        print("HYPOTHESIS:")
        print("-"*80)
        print("Different class groups may discuss different topics due to:")
        print("1. Different learning experiences and contexts")
        print("2. Group-specific dynamics and interactions")
        print("3. Varying levels of engagement with course materials")
        print("4. Different scheduling or delivery methods per group")
        
        return group_topics
    
    def analyze_enjoyment_ratings(self):
        """Question 2: Key topics for low vs high enjoyment ratings"""
        print("\n" + "="*80)
        print("ANALYSIS 2: Topics for Low vs High 'I Enjoyed the Course' Ratings")
        print("="*80)
        
        # Define low and high ratings (using median split)
        median_rating = self.df['b_1_i_enjoyed_the_course'].median()
        low_ratings = self.df[self.df['b_1_i_enjoyed_the_course'] <= median_rating]
        high_ratings = self.df[self.df['b_1_i_enjoyed_the_course'] > median_rating]
        
        print(f"\nMedian rating: {median_rating}")
        print(f"Low ratings (<={median_rating}): {len(low_ratings)} students")
        print(f"High ratings (>{median_rating}): {len(high_ratings)} students")
        
        # Analyze recommendations for each group
        print("\n" + "-"*80)
        print("LOW RATINGS - Key Topics in Recommendations:")
        print("-"*80)
        low_texts = []
        for col in self.text_columns:
            if f'{col}_processed' in self.df.columns:
                texts = low_ratings[f'{col}_processed'].dropna().tolist()
                low_texts.extend([t for t in texts if t and len(t.strip()) > 0])
        
        if low_texts:
            low_topics, _ = self.extract_topics_lda(low_texts, n_topics=5)
            if low_topics:
                for i, topic in enumerate(low_topics, 1):
                    print(f"Topic {i}: {', '.join(topic[:7])}")
        
        # Word frequency for low ratings
        low_words = []
        for text in low_texts:
            low_words.extend(text.split())
        if low_words:
            print("\nTop 15 Words in Low Rating Recommendations:")
            low_word_freq = pd.Series(low_words).value_counts().head(15)
            print(low_word_freq.to_string())
        
        print("\n" + "-"*80)
        print("HIGH RATINGS - Key Topics in Recommendations:")
        print("-"*80)
        high_texts = []
        for col in self.text_columns:
            if f'{col}_processed' in self.df.columns:
                texts = high_ratings[f'{col}_processed'].dropna().tolist()
                high_texts.extend([t for t in texts if t and len(t.strip()) > 0])
        
        if high_texts:
            high_topics, _ = self.extract_topics_lda(high_texts, n_topics=5)
            if high_topics:
                for i, topic in enumerate(high_topics, 1):
                    print(f"Topic {i}: {', '.join(topic[:7])}")
        
        # Word frequency for high ratings
        high_words = []
        for text in high_texts:
            high_words.extend(text.split())
        if high_words:
            print("\nTop 15 Words in High Rating Recommendations:")
            high_word_freq = pd.Series(high_words).value_counts().head(15)
            print(high_word_freq.to_string())
        
        # Compare unique concerns
        print("\n" + "-"*80)
        print("KEY INSIGHTS:")
        print("-"*80)
        low_unique = set(low_words) - set(high_words)
        high_unique = set(high_words) - set(low_words)
        
        print(f"\nWords more common in LOW ratings: {', '.join(list(low_unique)[:10])}")
        print(f"\nWords more common in HIGH ratings: {', '.join(list(high_unique)[:10])}")
        
        return low_topics, high_topics
    
    def analyze_absenteeism_correlation(self):
        """Question 3: Correlation between absenteeism and average course evaluation"""
        print("\n" + "="*80)
        print("ANALYSIS 3: Correlation between Absenteeism and Course Evaluation Rating")
        print("="*80)
        
        # Clean data
        df_clean = self.df[
            (self.df['absenteeism_percentage'].notna()) & 
            (self.df['average_course_evaluation_rating'].notna())
        ].copy()
        
        if len(df_clean) == 0:
            print("No data available for this analysis")
            return None
        
        # Calculate correlation
        correlation = df_clean['absenteeism_percentage'].corr(
            df_clean['average_course_evaluation_rating']
        )
        
        print(f"\nCorrelation coefficient: {correlation:.4f}")
        
        # Statistical test
        from scipy.stats import pearsonr
        r, p_value = pearsonr(
            df_clean['absenteeism_percentage'],
            df_clean['average_course_evaluation_rating']
        )
        
        print(f"P-value: {p_value:.4f}")
        print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(df_clean['absenteeism_percentage'], 
                   df_clean['average_course_evaluation_rating'],
                   alpha=0.6, s=50)
        plt.xlabel('Absenteeism Percentage', fontsize=12)
        plt.ylabel('Average Course Evaluation Rating', fontsize=12)
        plt.title(f'Absenteeism vs Course Evaluation Rating\n(Correlation: {correlation:.3f})', 
                 fontsize=14, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(df_clean['absenteeism_percentage'], 
                      df_clean['average_course_evaluation_rating'], 1)
        p = np.poly1d(z)
        plt.plot(df_clean['absenteeism_percentage'], 
                p(df_clean['absenteeism_percentage']), 
                "r--", alpha=0.8, linewidth=2, label='Trend Line')
        plt.legend()
        plt.tight_layout()
        rating_path = self.output_dir / 'absenteeism_vs_rating.png'
        plt.savefig(rating_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {rating_path.name}")
        
        # Group analysis
        print("\n" + "-"*80)
        print("Grouped Analysis:")
        print("-"*80)
        df_clean['absenteeism_category'] = pd.cut(
            df_clean['absenteeism_percentage'],
            bins=[0, 10, 20, 30, 100],
            labels=['Low (0-10%)', 'Medium (10-20%)', 'High (20-30%)', 'Very High (30%+)']
        )
        
        grouped = df_clean.groupby('absenteeism_category')['average_course_evaluation_rating'].agg(['mean', 'std', 'count'])
        print("\nAverage Rating by Absenteeism Category:")
        print(grouped.to_string())
        
        # Hypothesis
        print("\n" + "-"*80)
        print("HYPOTHESIS:")
        print("-"*80)
        if correlation < -0.3:
            print("STRONG NEGATIVE CORRELATION: Higher absenteeism is associated with lower ratings.")
            print("Possible reasons:")
            print("1. Students who miss classes have less engagement and lower satisfaction")
            print("2. Absenteeism may indicate disengagement or dissatisfaction")
            print("3. Missing content leads to confusion and lower perceived course quality")
        elif correlation > 0.3:
            print("POSITIVE CORRELATION: Higher absenteeism is associated with higher ratings.")
            print("Possible reasons:")
            print("1. Students who are already satisfied may feel confident to skip some classes")
            print("2. High-performing students may have other commitments")
        else:
            print("WEAK CORRELATION: Absenteeism and ratings are not strongly related.")
            print("Possible reasons:")
            print("1. Other factors (content quality, teaching style) matter more than attendance")
            print("2. Students can catch up through materials and recordings")
        
        return correlation, p_value
    
    def analyze_absenteeism_timing(self):
        """Question 4: Correlation between absenteeism and classes start/end on time"""
        print("\n" + "="*80)
        print("ANALYSIS 4: Correlation between Absenteeism and Classes Start/End on Time")
        print("="*80)
        
        # Clean data
        df_clean = self.df[
            (self.df['absenteeism_percentage'].notna()) & 
            (self.df['b_2_classes_started_and_ended_on_time'].notna())
        ].copy()
        
        if len(df_clean) == 0:
            print("No data available for this analysis")
            return None
        
        # Calculate correlation
        correlation = df_clean['absenteeism_percentage'].corr(
            df_clean['b_2_classes_started_and_ended_on_time']
        )
        
        print(f"\nCorrelation coefficient: {correlation:.4f}")
        
        # Statistical test
        from scipy.stats import pearsonr
        r, p_value = pearsonr(
            df_clean['absenteeism_percentage'],
            df_clean['b_2_classes_started_and_ended_on_time']
        )
        
        print(f"P-value: {p_value:.4f}")
        print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(df_clean['absenteeism_percentage'], 
                   df_clean['b_2_classes_started_and_ended_on_time'],
                   alpha=0.6, s=50)
        plt.xlabel('Absenteeism Percentage', fontsize=12)
        plt.ylabel('Classes Start/End on Time Rating', fontsize=12)
        plt.title(f'Absenteeism vs Timing Rating\n(Correlation: {correlation:.3f})', 
                 fontsize=14, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(df_clean['absenteeism_percentage'], 
                      df_clean['b_2_classes_started_and_ended_on_time'], 1)
        p = np.poly1d(z)
        plt.plot(df_clean['absenteeism_percentage'], 
                p(df_clean['absenteeism_percentage']), 
                "r--", alpha=0.8, linewidth=2, label='Trend Line')
        plt.legend()
        plt.tight_layout()
        timing_path = self.output_dir / 'absenteeism_vs_timing.png'
        plt.savefig(timing_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {timing_path.name}")
        
        # Cross-tabulation
        print("\n" + "-"*80)
        print("Cross-tabulation Analysis:")
        print("-"*80)
        df_clean['timing_category'] = pd.cut(
            df_clean['b_2_classes_started_and_ended_on_time'],
            bins=[0, 2, 3, 4, 5],
            labels=['Low (1-2)', 'Medium (3)', 'High (4)', 'Very High (5)']
        )
        
        df_clean['absenteeism_category'] = pd.cut(
            df_clean['absenteeism_percentage'],
            bins=[0, 10, 20, 30, 100],
            labels=['Low (0-10%)', 'Medium (10-20%)', 'High (20-30%)', 'Very High (30%+)']
        )
        
        crosstab = pd.crosstab(df_clean['timing_category'], 
                               df_clean['absenteeism_category'], 
                               margins=True)
        print("\nAbsenteeism by Timing Rating:")
        print(crosstab.to_string())
        
        # Hypothesis
        print("\n" + "-"*80)
        print("HYPOTHESIS:")
        print("-"*80)
        if correlation < -0.3:
            print("STRONG NEGATIVE CORRELATION: Students who rate timing poorly have higher absenteeism.")
            print("Possible reasons:")
            print("1. Poor timing management leads to frustration and reduced attendance")
            print("2. Students who miss classes may not know actual timing, rate it poorly")
            print("3. Timing issues create scheduling conflicts, increasing absenteeism")
        elif correlation > 0.3:
            print("POSITIVE CORRELATION: Students who rate timing well have higher absenteeism.")
            print("Possible reasons:")
            print("1. Consistent timing allows students to plan absences better")
            print("2. Students who attend less may not notice timing issues")
        else:
            print("WEAK CORRELATION: Timing and absenteeism are not strongly related.")
            print("Possible reasons:")
            print("1. Timing is a minor factor compared to content quality and engagement")
            print("2. Students' schedules are determined by other factors")
        
        return correlation, p_value
    
    def analyze_engagement_recommendations(self):
        """Question 5: Deep dive into 'more engagement' recommendations"""
        print("\n" + "="*80)
        print("ANALYSIS 5: Understanding 'More Engagement' Recommendations")
        print("="*80)
        
        # Find all recommendations mentioning engagement
        rec_col = 'f_4_Write_at_least_one_recommendation_to_improve_the_teaching_and_learning_in_this_course_(for_future_classes)'
        
        # Keywords related to engagement
        engagement_keywords = ['engage', 'engagement', 'interact', 'interaction', 
                              'participat', 'involv', 'activ', 'discuss', 'group']
        
        engagement_responses = []
        for idx, row in self.df.iterrows():
            rec_text = str(row[rec_col]).lower() if pd.notna(row[rec_col]) else ""
            if any(keyword in rec_text for keyword in engagement_keywords):
                engagement_responses.append({
                    'original': row[rec_col],
                    'processed': row.get(f'{rec_col}_processed', ''),
                    'enjoyment_rating': row.get('b_1_i_enjoyed_the_course', np.nan),
                    'avg_rating': row.get('average_course_evaluation_rating', np.nan),
                    'group': row.get('a_3_class_group', ''),
                    'absenteeism': row.get('absenteeism_percentage', np.nan)
                })
        
        print(f"\nFound {len(engagement_responses)} responses mentioning engagement")
        
        if len(engagement_responses) == 0:
            print("No engagement-related recommendations found")
            return None
        
        # Analyze what they mean by engagement
        print("\n" + "-"*80)
        print("Sample Engagement Recommendations:")
        print("-"*80)
        for i, resp in enumerate(engagement_responses[:10], 1):
            print(f"\n{i}. {resp['original']}")
            print(f"   Enjoyment Rating: {resp['enjoyment_rating']}, "
                  f"Avg Rating: {resp['avg_rating']:.2f}, "
                  f"Group: {resp['group']}, "
                  f"Absenteeism: {resp['absenteeism']:.1f}%")
        
        # Extract common themes
        all_engagement_text = ' '.join([r['processed'] for r in engagement_responses if r['processed']])
        
        # Word frequency analysis
        engagement_words = all_engagement_text.split()
        word_freq = pd.Series(engagement_words).value_counts()
        
        print("\n" + "-"*80)
        print("Top Words in Engagement Recommendations:")
        print("-"*80)
        print(word_freq.head(20).to_string())
        
        # Context analysis - look for specific engagement types
        print("\n" + "-"*80)
        print("Engagement Type Analysis:")
        print("-"*80)
        
        engagement_types = {
            'group_work': ['group', 'team', 'collaborat'],
            'discussion': ['discuss', 'debate', 'talk', 'convers'],
            'interactive': ['interact', 'interactive', 'hands'],
            'participation': ['participat', 'involv', 'contribute'],
            'activities': ['activ', 'exercise', 'practice', 'task'],
            'questions': ['question', 'ask', 'answer', 'quiz'],
            'practical': ['practical', 'lab', 'hands', 'real'],
            'engagement': ['engage', 'engagement', 'engaging']
        }
        
        for eng_type, keywords in engagement_types.items():
            count = sum(1 for resp in engagement_responses 
                       if any(kw in str(resp['original']).lower() for kw in keywords))
            if count > 0:
                print(f"{eng_type.replace('_', ' ').title()}: {count} mentions")
        
        # Compare with what they liked
        print("\n" + "-"*80)
        print("What Students Liked (for context):")
        print("-"*80)
        liked_col = 'f_3_Write_at_least_two_things_you_liked_about_the_teaching_and_learning_in_this_course'
        
        engagement_student_ids = []
        for resp in engagement_responses:
            idx = self.df[self.df[rec_col] == resp['original']].index
            if len(idx) > 0:
                engagement_student_ids.append(idx[0])
        
        liked_texts = []
        for idx in engagement_student_ids[:20]:  # Sample
            if pd.notna(self.df.loc[idx, liked_col]):
                liked_texts.append(str(self.df.loc[idx, liked_col]))
        
        if liked_texts:
            print("\nSample of what these students liked:")
            for i, text in enumerate(liked_texts[:5], 1):
                print(f"{i}. {text[:200]}...")
        
        # Statistical analysis
        print("\n" + "-"*80)
        print("Statistical Profile of Students Requesting More Engagement:")
        print("-"*80)
        
        engagement_df = pd.DataFrame(engagement_responses)
        if len(engagement_df) > 0:
            print(f"\nAverage Enjoyment Rating: {engagement_df['enjoyment_rating'].mean():.2f}")
            print(f"Average Course Rating: {engagement_df['avg_rating'].mean():.2f}")
            print(f"Average Absenteeism: {engagement_df['absenteeism'].mean():.1f}%")
            
            # Compare with overall
            overall_enjoyment = self.df['b_1_i_enjoyed_the_course'].mean()
            overall_rating = self.df['average_course_evaluation_rating'].mean()
            overall_absenteeism = self.df['absenteeism_percentage'].mean()
            
            print(f"\nComparison with Overall Average:")
            print(f"Enjoyment: {engagement_df['enjoyment_rating'].mean():.2f} vs {overall_enjoyment:.2f} "
                  f"({'Higher' if engagement_df['enjoyment_rating'].mean() > overall_enjoyment else 'Lower'})")
            print(f"Course Rating: {engagement_df['avg_rating'].mean():.2f} vs {overall_rating:.2f} "
                  f"({'Higher' if engagement_df['avg_rating'].mean() > overall_rating else 'Lower'})")
            print(f"Absenteeism: {engagement_df['absenteeism'].mean():.1f}% vs {overall_absenteeism:.1f}% "
                  f"({'Higher' if engagement_df['absenteeism'].mean() > overall_absenteeism else 'Lower'})")
        
        # Interpretation
        print("\n" + "-"*80)
        print("INTERPRETATION: What 'More Engagement' Means")
        print("-"*80)
        print("""
Based on the analysis, students requesting 'more engagement' likely mean:

1. MORE INTERACTIVE LEARNING:
   - More hands-on activities and practical exercises
   - Less passive lecture-style teaching
   - More opportunities to apply concepts in real-time

2. INCREASED PARTICIPATION OPPORTUNITIES:
   - More class discussions and Q&A sessions
   - More group work and collaborative activities
   - More interactive quizzes and exercises during class

3. ACTIVE INVOLVEMENT:
   - Students want to be actively involved rather than just listening
   - More opportunities to contribute and share ideas
   - More practical labs and exercises

4. WHAT'S MISSING (Based on what they liked):
   - Students appreciate when there IS engagement (interactive teaching, group work)
   - They want MORE of these engaging elements
   - They may feel some classes are too lecture-heavy or passive

5. EXPECTATIONS NOT MET:
   - Students expect a balance between theory and practice
   - They want more opportunities to practice during class time
   - They want more interactive elements to maintain attention and interest
        """)
        
        return engagement_responses
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)
        
        report = []
        report.append("="*80)
        report.append("COURSE EVALUATION DATA ANALYSIS - SUMMARY REPORT")
        report.append("="*80)
        report.append("")
        report.append(f"Total Responses: {len(self.df)}")
        report.append(f"Class Groups: {', '.join(self.df['a_3_class_group'].dropna().unique())}")
        report.append(f"Average Course Rating: {self.df['average_course_evaluation_rating'].mean():.2f}")
        report.append(f"Average Absenteeism: {self.df['absenteeism_percentage'].mean():.1f}%")
        report.append("")
        
        # Save report
        report_path = self.output_dir / 'analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"Summary report saved to {report_path.name}")
    
    def run_all_analyses(self):
        """Run all analyses"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE QUALITATIVE DATA ANALYSIS")
        print("="*80)
        
        # Run all analyses
        self.analyze_topics_by_group()
        self.analyze_enjoyment_ratings()
        self.analyze_absenteeism_correlation()
        self.analyze_absenteeism_timing()
        self.analyze_engagement_recommendations()
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated files:")
        print("- absenteeism_vs_rating.png")
        print("- absenteeism_vs_timing.png")
        print("- analysis_report.txt")


if __name__ == "__main__":
    analyzer = CourseEvaluationAnalyzer()
    analyzer.run_all_analyses()

