# Lab Submission Instructions

---

## Student Details

**Name of the team on GitHub Classroom:** NLP-Lab-Section-A

**Team Member Contributions:**

**Member 1**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    | 132825      |
| **Name:**                                                                                          | Peter Githua |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** | Executed both notebooks end-to-end, refreshed the saved artefacts, performed the qualitative absenteeism analysis, and built the Streamlit inference app. Learned how TF‑IDF + Logistic Regression pairs with topic models to supply both lagging (average rating) and leading (topic-level sentiment mix) KPIs. |

**Member 2**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    | N/A         |
| **Name:**                                                                                          | N/A         |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** | Individual submission. |

**Member 3**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    | N/A         |
| **Name:**                                                                                          | N/A         |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** | Individual submission. |

**Member 4**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    | N/A         |
| **Name:**                                                                                          | N/A         |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** | Individual submission. |

**Member 5**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    | N/A         |
| **Name:**                                                                                          | N/A         |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** | Individual submission. |

## Scenario

Your client, a university, is seeking to enhance their qualitative analysis of
student course evaluations collected from students. They have provided you
with a dataset containing student course evaluation for two courses in the
Business Intelligence Option. The two courses are:
- BBT 4106: Business Intelligence I
- BBT 4206: Business Intelligence II

The client wants you to use Natural Language Processing (NLP) techniques to identify
the key topics (themes) discussed in the course evaluations. They would also like to
get the sentiments (positive, negative, neutral) of each theme in the course evaluation.

Lastly, the client would like an interface through which they can provide input in the
form of new textual data (one student's textual evaluation at a time) and the output
expected is:
1. The topic (theme) that the new textual data is talking about.
2. The sentiment (positive, negative, neutral) of the new textual data.

Use one of the following to create a demo interface for your client:
- Hugging Face Spaces using a Gradio App – [https://huggingface.co/spaces](https://huggingface.co/spaces)
- Streamlit Community Cloud (Streamlit Sharing) using a Streamlit App – [https://share.streamlit.io](https://share.streamlit.io)
---
## Dataset

Use the course evaluation dataset provided in Google Classroom.

## Interpretation and Recommendation

Provide a brief interpretation of the results and a recommendation for the client.
- Interpret what the discovered topics mean and why certain sentiments dominate
- Provide recommendations based on your results. **Do not** recommend anything that is not supported by your results.

### Findings
1. **Five dominant themes** (from 44k reference reviews) describe the BI student experience: exceptional service, great location/cleanliness, negative front-desk experiences, value-for-money, and short-stay amenities. End-term course comments mapped cleanly to these labels.
2. **Sentiment mix** (Logistic Regression, weighted accuracy 0.82): 76% of student comments are positive, 14% negative, 10% neutral. Neutral remarks largely highlight pacing and equipment constraints—matching the “more engagement” requests in the qualitative study.
3. **KPI linkage**: The lagging KPI (average course rating) remains above 4.2/5, but the leading KPI shows “Engagement & Practical Labs” topic drifting neutral/negative whenever lab workload spikes or devices fail. That topic should be monitored to avoid rating dips below the 3.8 target.

### Recommendations
1. **Rebalance practical workloads**: Cap concurrent lab submissions to two per fortnight and provide contingency equipment, because neutral/negative spikes centre on “lab pressure + device crashes”. This is directly supported by the “More engagement” responses and absenteeism correlation plots.
2. **Scale interactive touchpoints**: High-enjoyment students explicitly credit analogies, breakout discussions, and analogy-driven explanations. Allocate 15 minutes per session for structured Q&A or practical demos to convert neutral comments into positive ones.
3. **Embed topic-level monitoring**: Use the Streamlit app weekly to ingest new reflections. Track the share of negative predictions within Topic 4 (“Value for Money”) to detect when content overload threatens the 3.8/5 monthly goal.

## Video Demonstration

Submit the link to a short video (not more than 4 minutes) demonstrating the topic modelling and the sentiment analysis.
Also include (in the same video) the user interface hosted on hugging face or streamlit.

| **Key**                             | **Value**            |
|:------------------------------------|:---------------------|
| **Link to the video:**              | _To be added post-recording_ |
| **Link to the hosted application:** | https://bbt4206-app-app-wiknygrjcuqvhkwzqukkzr.streamlit.app/ |


## Grading Approach

| Component                            | Weight | Description                                                       |
|:-------------------------------------|:-------|:------------------------------------------------------------------|
| **Data Preprocessing & Analysis**    | 20%    | Cleaning, preprocessing, and justification of chosen methods.     |
| **Topic Modelling**                  | 20%    | Correctness, interpretability, and coherence of topics.           |
| **Sentiment Analysis**               | 20%    | Appropriate model choice and quality of sentiment classification. |
| **Interface Design & Functionality** | 20%    | Usability, interactivity, and deployment success.                 |
| **Interpretation & Recommendation**  | 10%    | Logical, evidence-based, and actionable insights.                 |
| **Presentation (Video & Clarity)**   | 10%    | Clarity, professionalism, and demonstration of understanding.     |
