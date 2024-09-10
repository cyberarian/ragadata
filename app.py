import streamlit as st
import pandas as pd
import plotly.express as px
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import io
from PyPDF2 import PdfReader

# Set page config to wide mode
st.set_page_config(layout="wide")

# Get GitHub token from Streamlit secrets
token = st.secrets["github"]["token"]

# Azure inference endpoint and model
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

# Initialize the client with the token
client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

# Function to read PDF content
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to load data from uploaded file
def load_data(file):
    file_extension = file.name.split(".")[-1].lower()
    if file_extension == "csv":
        df = pd.read_csv(file)
    elif file_extension in ["xls", "xlsx"]:
        df = pd.read_excel(file)
    elif file_extension == "pdf":
        text = read_pdf(file)
        return text
    else:
        st.error("Unsupported file format. Please upload a CSV, XLSX, or PDF file.")
        return None
    return df

# Function to create plots based on user selection
def create_plot(df):
    if df is not None and isinstance(df, pd.DataFrame):
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) >= 2:
            plot_type = st.radio("Select Plot Type", ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram"])
            
            x_col = st.selectbox("Select X-axis", numeric_cols)
            y_col = st.selectbox("Select Y-axis", numeric_cols) if plot_type != "Histogram" else None

            if plot_type == "Scatter Plot":
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            elif plot_type == "Line Chart":
                fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            elif plot_type == "Bar Chart":
                fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            elif plot_type == "Histogram":
                fig = px.histogram(df, x=x_col, title=f"Histogram of {x_col}")

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Brief Insights"):
                if plot_type != "Histogram" and y_col:
                    correlation = df[x_col].corr(df[y_col])
                    st.write(f"Correlation between {x_col} and {y_col}: {correlation:.2f}")
                st.write(f"Summary statistics for {x_col}:")
                st.write(df[x_col].describe())
                if y_col:
                    st.write(f"Summary statistics for {y_col}:")
                    st.write(df[y_col].describe())
        else:
            st.warning("The dataframe doesn't have enough numeric columns for plotting.")
    else:
        st.warning("Please upload a valid CSV or XLSX file for visualization.")

# Sidebar
st.sidebar.image("ragadata2.png", use_column_width=True)  # Replace with your logo
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Guides", "Support"])

if page == "Home":
    st.title(":cat: Data Analysis, Visualization, and PDF Interaction")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "pdf"])

    if uploaded_file is not None:
        # Load the data
        data = load_data(uploaded_file)
        
        # Main content area
        main_container = st.container()
        
        with main_container:
            # Tabs
            tab1, tab2 = st.tabs(["Data Preview & Chat", "Data Visualization"])
            
            with tab1:
                if isinstance(data, pd.DataFrame):
                    st.subheader("Data Preview")
                    st.write(data.head())
                    
                    with st.expander("Data Statistics"):
                        st.write(f"Number of rows: {data.shape[0]}")
                        st.write(f"Number of columns: {data.shape[1]}")
                        st.write("Column types:")
                        st.write(data.dtypes)
                        st.write("Summary statistics:")
                        st.write(data.describe())
                        

                # Chat interface
                st.subheader("Ask a Question")
                user_input = st.text_input("Enter your question", "What insights can you provide from the uploaded data?")

                if st.button("Submit"):
                    # Prepare context from uploaded data
                    context = ""
                    if uploaded_file is not None:
                        if isinstance(data, pd.DataFrame):
                            context = f"Data summary:\n{data.describe().to_string()}\n\n"
                        elif isinstance(data, str):
                            context = f"PDF content summary:\n{data[:1000]}...\n\n"
                    
                    # Send the request to Azure AI model
                    response = client.complete(
                        messages=[
                            SystemMessage(content="You are a helpful assistant. Analyze the provided data and answer questions."),
                            UserMessage(content=f"{context}User question: {user_input}"),
                        ],
                        model=model_name,
                        temperature=0.7,
                        max_tokens=1000,
                        top_p=1.0
                    )

                    # Display the response in Streamlit
                    st.write("Response:")
                    st.write(response.choices[0].message.content)
            
            with tab2:
                st.subheader("Data Visualization")
                create_plot(data if isinstance(data, pd.DataFrame) else None)

# Footer
    st.markdown("---")
    st.markdown("Built with :orange_heart: thanks to Claude 3.5 Sonnet, Github Models, Streamlit. :scroll: support my works at https://saweria.co/adnuri", help="cyberariani@gmail.com")



elif page == "About":
    st.title("About RAGAData Chat")
    st.write("Introduction")
    st.markdown("""
    Welcome to RAGAData Chat, a user-friendly tool designed to simplify and enhance your data analysis experience. In today's data-driven world, having a reliable tool to help you understand and interpret your information is crucial, and that's exactly what we aim to provide with RAGAData Chat.

    Our tool supports a variety of file formats, including CSV, XLSX, and PDF, making it easy for you to upload your data and start exploring. RAGAData Chat is powered by OpenAI's GPT-4o, allowing for a more interactive and intuitive experience. You can visualize your data through straightforward charts and graphs, and engage with our AI to ask questions and gain insights directly from your datasets.

    In addition, we're fortunate to have access to the limited public beta for GitHub Models, which helps enhance the capabilities of our AI, allowing us to offer a more refined and useful experience.

    Whether you're analyzing data for research, business, or personal projects, RAGAData Chat is here to assist you with a clear and accessible approach to data analysis. We hope this tool makes your work easier and more insightful.
    """)
    # Add more content about your application

elif page == "Guides":
    st.title("User Guides")
    st.write("Learn how to use RAGAData Chat effectively:")
    st.markdown("""
    1. **Uploading Data**: Click on 'Choose a file' to upload your CSV, XLSX, or PDF file.
    2. **Data Preview**: For CSV and XLSX files, you can see a preview of your data in the 'Data Analysis' tab.
    3. **Asking Questions**: Enter your question about the data and click 'Submit' to get AI-powered insights.
    4. **Data Visualization**: Switch to the 'Data Visualization' tab to create simple plots of your data.
    """)
    # Add more detailed guides as needed

elif page == "Support":
    st.title("Support")
    st.write("Thank you for using RAGAData Chat!")
    st.markdown("""
    I’m thrilled to have you on board, and your support means the world to me. If you’ve enjoyed using RAGAData Chat and find it helpful in your data analysis work, there are a few ways you can support me to improve and sustain this project.

    1. Star the Project on GitHub\n
    One of the easiest yet most impactful ways to show your support is by giving RAGAData Chat a star on GitHub. Your star helps increase the project's visibility and shows others that it’s a tool worth exploring. Simply visit the GitHub page and hit the star button—it’s quick and easy, but it goes a long way in helping the project grow!

    2. Support Through Donations\n
    I am continually working on new features, improving performance, and enhancing the overall experience of RAGAData Chat. To keep this project alive and growing, I rely on the generosity of users like you. I also aim to build a high-end computer to handle the computing demands of AI, which are resource-intensive. If you'd like to help me add more features and make RAGAData Chat even better, consider making a donation. You can support me through:

    Saweria.co: For users in Indonesia, you can donate via  https://saweria.co/adnuri, a local crowdfunding platform.\n
    Your support will help cover development costs, including computing resources, tools, and infrastructure, ensuring that RAGAData Chat can continue to evolve and perform at its best.

    3. Get in Touch\n
    If you have any questions, suggestions, or feedback, feel free to reach out to me directly. I’d love to hear from you! You can contact me at cyberariani@gmail.com. Whether you’re interested in new features or just want to share your experience, your input will help shape the future of RAGAData Chat.

    Once again, thank you for your support. I look forward to making RAGAData Chat even better with your help!
    
    """)
    # Add a contact form or more support information
    
    