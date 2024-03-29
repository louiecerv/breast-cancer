import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():
    if "reset_app" not in st.session_state:
        st.session_state.reset_app = False

    text = """ Logistic Regression, Decision Tree, SVM, KNN and Naive Bayes on the Breast Cancer Dataset"""
    st.subheader(text)

    # Use session state to track the current form
    if "current_form" not in st.session_state:
        st.session_state["current_form"] = 1    

    # Display the appropriate form based on the current form state
    if st.session_state["current_form"] == 1:
        display_form1()
    elif st.session_state["current_form"] == 2:
        display_form2()
    elif st.session_state["current_form"] == 3:
        display_form3()

    if "clf" not in st.session_state: 
        st.session_state["clf"] = []

    if "X_train" not in st.session_state: 
        st.session_state["X_train"] = []

    if "X_test" not in st.session_state: 
        st.session_state["X_test"] = []
    
    if "y_train" not in st.session_state: 
        st.session_state["X_train"] = []
    
    if "y_test" not in st.session_state: 
        st.session_state["y_yest"] = []

    if "selected_model" not in st.session_state: 
        st.session_state["selected_model"] = 0

def display_form1():
    st.session_state["current_form"] = 1
    form1 = st.form("intro")

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    form1.text(text)

    form1.subheader('Breast Cancer Diagnosis Support Using ML Classifiers')
    form1.image('breast-cancer.jpg', caption="The Breast Cancer Dataset in Sci-kit Learn", use_column_width=True)
    text = """The breast cancer dataset in scikit-learn is a well-known dataset used for binary 
        classification tasks. It contains data collected from patients diagnosed with breast cancer. 
        Here's a breakdown of the dataset:
        \nSource: The data consists of features extracted from digitized images of fine needle 
        aspirates (FNA) of breast masses.
        \nFeatures: The dataset includes 569 data points, each with 30 numerical features. 
        These features represent various characteristics of the cell nuclei, such as radius, 
        texture, perimeter, and area.
        \nTarget: The target variable indicates the class, whether the tumor is malignant 
        (cancerous) or benign (non-cancerous). There are 212 malignant cases and 357 benign cases."""
    form1.write(text)
  
    submit1 = form1.form_submit_button("Start")

    if submit1:
        # Go to the next form        
        display_form2()

def display_form2():
    st.session_state["current_form"] = 2

    form2 = st.form("training")
    # Load the iris dataset
    data = load_breast_cancer()

    # Convert data features to a DataFrame
    feature_names = data.feature_names
    df = pd.DataFrame(data.data, columns=feature_names)
    df['target'] = data.target
    
    # Separate features and target variable
    X = df.drop('target', axis=1)  # Target variable column name
    y = df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #save the values to the session state
    
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test

    form2.subheader('Browse the Dataset') 
    form2.write(df)

    form2.subheader('Dataset Description')
    form2.write(df.describe().T)

    # Create the plot using seaborn.countplot
    fig, ax = plt.subplots(figsize=(6, 6))  # Set appropriate figure size
    sns.countplot(x="target", data=df, ax=ax, hue="target", palette="bright")  # Plot with color encoding

    # Customize the plot for clarity and aesthetics
    plt.xlabel("Target")
    plt.ylabel("Number of Samples")
    plt.title("Distribution of Target Classes in Breast Cancer Dataset")
    plt.xticks([0, 1], labels=data.target_names)  # Use informative labels for target classes
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add subtle grid lines

    # Display the plot
    plt.tight_layout()
    form2.pyplot(fig)

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    form2.pyplot(fig)

    form2.subheader('Select the classifier')

    # Create the selecton of classifier
    clf = LogisticRegression(random_state=42)
    st.session_state['selected_model'] = 0
    options = ['Logistic Regression', 'Support Vector Machine', 'K-Nearest Neighbor', 'Naive Bayes']
    selected_option = form2.selectbox('Select the classifier', options)
    if selected_option=='Support Vector Machine':        
        clf = SVC(kernel='linear', random_state=42)
        st.session_state['selected_model'] = 1
    elif selected_option=='K-Nearest Neighbor':        
        clf = KNeighborsClassifier(n_neighbors=5)
        st.session_state['selected_model'] = 2
    elif selected_option=='Naive Bayes':        
        clf = GaussianNB()
        st.session_state['selected_model'] = 3
    else:
        clf = LogisticRegression(random_state=42)
        st.session_state['selected_model'] = 0

    # save the clf to the session variable
    st.session_state['clf'] = clf

    submit2 = form2.form_submit_button("Train")
    if submit2:     
        display_form3()

def display_form3():
    st.session_state["current_form"] = 3
    form3 = st.form("Result")
    classifier = ''
    if st.session_state['selected_model'] == 0:     # decision tree
        report = """Logistic regression generally performs well on the breast 
        cancer dataset in scikit-learn, achieving high accuracy scores 
        (often exceeding 95%) in various studies and implementations. 
        This indicates the model's effectiveness in distinguishing 
        between malignant and benign tumors based on the provided features."""
        classifier = 'Logistic Regression'
    elif st.session_state['selected_model'] == 1:   
        report = """Support vector machines (SVMs) demonstrate strong
        performance on the breast cancer dataset in scikit-learn. 
        They often achieve accuracy scores comparable or exceeding 95%, 
        showcasing their capability to effectively separate malignant and 
        benign tumors based on the given features."""
        classifier = 'Support Vector Machine'
    elif st.session_state['selected_model'] == 2:   
        report = """The performance of KNN on the breast cancer dataset in 
        scikit-learn is generally positive, achieving decent accuracy 
        scores (often around 90% or higher). However, it's important to
        understand some nuances: Performance is sensitive to the "k"
        parameter: This parameter determines the number of nearest 
        neighbors considered for classification. Choosing the optimal 
        "k" value requires careful tuning as both too high and too 
        low values can negatively impact accuracy."""
        classifier = "K-Nearest Neighbor"
    else:
        report = """Similar to logistic regression, Naive Bayes also exhibits
        good performance on the breast cancer dataset in scikit-learn. 
        Studies and implementations often report accuracy scores 
        exceeding 90%, demonstrating its ability to effectively 
        classify malignant and benign tumors."""
        classifier = "Naive Bayes"

    form3.subheader('Performance of the ' + classifier)

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train= st.session_state['y_train']
    y_test = st.session_state['y_test']

    clf = st.session_state['clf']
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)

    form3.subheader('Confusion Matrix')
    cm = confusion_matrix(y_test, y_test_pred)
    form3.text(cm)    
    text = """This is a table with two rows and two columns, representing the actual classes (benign/malignant)
        and the predicted classes by the model. The four values in the table represent:
        \nTrue Positive (TP): These are the samples where the model correctly predicted a malignant tumor.
        \nTrue Negative (TN): These are the samples where the model correctly predicted a benign tumor.
        \nFalse Positive (FP): These are the benign tumors incorrectly classified as malignant by the model 
        (Type I error). This can be a critical error in cancer diagnosis.
        \nFalse Negative (FN): These are the malignant tumors incorrectly classified as benign by the model 
        (Type II error). This is an even more critical error as it can lead to delayed or missed treatment.
        \nBy analyzing the confusion matrix, you can gain insights beyond just accuracy. You can identify:
        \nOverall Performance: The diagonal (TP and TN) represents correct predictions, indicating the 
        model's general ability to classify tumors.
        \nClass Imbalance: If there's a significant difference between the number of benign and malignant 
        tumors, the confusion matrix can reveal any bias towards the majority class.
        \nFocus Areas: Depending on the application, you might prioritize minimizing either False Positives 
        or False Negatives. The confusion matrix helps you identify which type of error your model 
        struggles with more."""
    form3.write(text)

    form3.subheader('Performance Metrics')
    form3.text(classification_report(y_test, y_test_pred))
    form3.write(report)

    # save the clf to the session state
    st.session_state['clf'] = clf
    submit3 = form3.form_submit_button("Reset")
    if submit3:     
        display_form1()

if __name__ == "__main__":
    app()
