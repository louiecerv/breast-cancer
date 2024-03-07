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

    form1.subheader('Breast Cancer Diagnosis')
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
    
    form2.write('The breast cancer dataset')
    form2.write(df)

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

    fig, ax = plt.subplots()
    sns.countplot(x="target", data=df, ax=ax)
    form2.pyplot(fig)

    # Number of features to visualize in pairplot (adjust as needed)
    num_features = 4

    # Create figure and subplots
    fig, axs = plt.subplots(nrows=int((num_features - 1) / 2) + 1, ncols=2)

    # Pairplot using a loop (adjust slice for desired features)
    for i in range(num_features):
        for j in range(i + 1, num_features):
            sns.kdeplot(df[data.feature_names[i]], df[data.feature_names[j]], hue="target", ax=axs[int(i // 2), i % 2])

    form2.pyplot(fig)

    fig, ax = plt.subplots()
    sns.displot(df["mean area"], hue="target", kind="kde", ax=ax)
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
        text = """Logistic regression generally performs well on the breast 
        cancer dataset in scikit-learn, achieving high accuracy scores 
        (often exceeding 95%) in various studies and implementations. 
        This indicates the model's effectiveness in distinguishing 
        between malignant and benign tumors based on the provided features."""
        classifier = 'Logistic Regression'
    elif st.session_state['selected_model'] == 1:   
        text = """Support vector machines (SVMs) also demonstrate strong
        performance on the breast cancer dataset in scikit-learn. 
        They often achieve accuracy scores comparable or exceeding 95%, 
        showcasing their capability to effectively separate malignant and 
        benign tumors based on the given features."""
        classifier = 'Support Vector Machine'
    elif st.session_state['selected_model'] == 2:   
        text = """The performance of KNN on the breast cancer dataset in 
        scikit-learn is generally positive, achieving decent accuracy 
        scores (often around 90% or higher). However, it's important to
        understand some nuances: Performance is sensitive to the "k"
        parameter: This parameter determines the number of nearest 
        neighbors considered for classification. Choosing the optimal 
        "k" value requires careful tuning as both too high and too 
        low values can negatively impact accuracy."""
        classifier = "K-Nearest Neighbor"
    else:
        text = """Similar to logistic regression, Naive Bayes also exhibits
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
    form3.write('Confusion Matrix')
    cm = confusion_matrix(y_test, y_test_pred)
    form3.text(cm)

    form3.subheader('Performance Metrics')
    form3.text(classification_report(y_test, y_test_pred))

    form3.write(text)

    # save the clf to the session state
    st.session_state['clf'] = clf

    submit3 = form3.form_submit_button("Reset")
    if submit3:
        st.session_state.reset_app = True
        st.session_state.clear()
        form3.write("If the form does not reset, click the reset button again.")

if __name__ == "__main__":
    app()
