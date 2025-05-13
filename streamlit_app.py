import streamlit as st
import pandas as pd
import os
import uuid
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Try to import the required modules
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    st.error("""
    ❌ Error: The 'google.generativeai' package could not be imported. 
    
    This may be caused by:
    1. The package is not installed: Run `pip install google-generativeai==0.8.5`
    2. You're using a different Python environment than where the package was installed
    
    AI agent functionality will be disabled until this is fixed.
    """)

# Try to import the agent classes from the main module
try:
    from data_analysis_a2a import (
        DataAnalyst, DataScientist, DataVisualizationAnalyst, DataStoryteller
    )
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False
    st.error(f"""
    ❌ Error: Could not import agent classes from data_analysis_a2a.py.
    
    Error details: {str(e)}
    
    This may be caused by:
    1. The file is missing or has been moved
    2. There are syntax errors in the file
    
    Please ensure data_analysis_a2a.py is in the same directory as this file.
    """)

# Set page configuration
st.set_page_config(
    page_title="A2A Data Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Initialize session state for API key
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False
    # Only check if API key is valid if GENAI_AVAILABLE is True
    if GENAI_AVAILABLE and st.session_state.api_key:
        try:
            genai.configure(api_key=st.session_state.api_key)
            model = genai.GenerativeModel(model_name="gemini-pro")
            model.generate_content("Hello")
            st.session_state.api_key_valid = True
        except Exception:
            st.session_state.api_key_valid = False

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "df" not in st.session_state:
    st.session_state.df = None
if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize agents if available
if AGENTS_AVAILABLE:
    if "agents" not in st.session_state:
        st.session_state.agents = {
            "analyst": DataAnalyst(),
            "scientist": DataScientist(),
            "visualizer": DataVisualizationAnalyst(),
            "storyteller": DataStoryteller()
        }

# App title and description
st.title("A2A Data Analysis System")
st.markdown("*Powered by Google Gemini Pro*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("API Configuration")
    api_key_input = st.text_input(
        "Enter your Gemini API key:",
        value=st.session_state.api_key,
        type="password",
        help="Get your API key from https://ai.google.dev/"
    )
    
    # Update API key if changed
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        try:
            if st.session_state.api_key and GENAI_AVAILABLE:
                # Test the API key
                genai.configure(api_key=st.session_state.api_key)
                model = genai.GenerativeModel(model_name="gemini-pro")
                model.generate_content("Hello")
                st.session_state.api_key_valid = True
                st.success("API key is valid!")
            else:
                st.session_state.api_key_valid = False
                if not GENAI_AVAILABLE:
                    st.warning("Cannot validate API key: google.generativeai module is not available")
        except Exception as e:
            st.session_state.api_key_valid = False
            st.error(f"Invalid API key: {str(e)}")
    
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            # Load the data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(temp_path)
            else:
                df = pd.read_excel(temp_path)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.file_path = temp_path
            st.session_state.file_name = uploaded_file.name
            
            st.success(f"Uploaded: {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            
    st.header("Data Agents")
    agent_info = {
        "analyst": "Analyzes data and provides statistical insights",
        "scientist": "Builds and evaluates machine learning models",
        "visualizer": "Creates visualizations and charts",
        "storyteller": "Creates narratives from data insights"
    }
    
    for agent_key, description in agent_info.items():
        st.subheader(f"{agent_key.title()}")
        st.markdown(f"*{description}*")

# Main content area
if not GENAI_AVAILABLE:
    st.warning("""
    AI functionality is disabled because the google.generativeai module is not available.
    
    To fix this:
    1. Open a terminal or command prompt
    2. Run: `pip install google-generativeai==0.8.5`
    3. Restart this application
    
    You can still use basic data analysis features without AI functionality.
    """)
elif not st.session_state.api_key_valid:
    st.warning("Please enter a valid Gemini API key in the sidebar to use AI functionalities.")
    st.info("You can get a free API key from [Google AI Studio](https://ai.google.dev/)")
    
if st.session_state.df is None:
    st.info("Please upload a CSV or Excel file to begin analysis.")
else:
    # Display data overview
    st.header("Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Information")
        st.write(f"Filename: {st.session_state.file_name}")
        st.write(f"Rows: {st.session_state.df.shape[0]}")
        st.write(f"Columns: {st.session_state.df.shape[1]}")
    
    with col2:
        st.subheader("Column Information")
        column_df = pd.DataFrame({
            "Type": st.session_state.df.dtypes,
            "Non-Null Count": st.session_state.df.count(),
            "Null Count": st.session_state.df.isna().sum(),
            "Unique Values": st.session_state.df.nunique()
        })
        st.dataframe(column_df)
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(st.session_state.df.head())
    
    # Statistical Description
    st.subheader("Statistical Description")
    st.dataframe(st.session_state.df.describe().T)
    
    # Chat with Agents
    st.header("Chat with Data Agents")
    
    if not AGENTS_AVAILABLE:
        st.warning("Agent functionality is unavailable because agent classes could not be imported.")
    elif not st.session_state.api_key_valid:
        st.warning("Please enter a valid Gemini API key to use the AI agents.")
    elif not GENAI_AVAILABLE:
        st.warning("AI agent functionality is disabled because the google.generativeai module could not be imported.")
    else:
        # Agent selection
        agent_type = st.selectbox(
            "Select an agent to work with:",
            list(st.session_state.agents.keys()),
            format_func=lambda x: x.title()
        )
        
        # Display agent description
        st.markdown(f"*{agent_info[agent_type]}*")
        
        # Input for query
        user_query = st.text_area("Enter your query for the agent:", height=100)
        
        # Process query
        if st.button("Submit Query", type="primary"):
            if not user_query:
                st.warning("Please enter a query.")
            else:
                with st.spinner(f"Processing query with {agent_type.title()}..."):
                    try:
                        # Configure API with the user-provided key
                        if GENAI_AVAILABLE:
                            genai.configure(api_key=st.session_state.api_key)
                        
                        # Add user message to chat history
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": user_query,
                            "agent": agent_type
                        })
                        
                        # Process the query
                        response = st.session_state.agents[agent_type].process(
                            user_input=user_query,
                            df=st.session_state.df,
                            api_key=st.session_state.api_key
                        )
                        
                        # Add assistant message to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "agent": agent_type
                        })
                        
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
    
    # Display chat history
    st.subheader("Conversation History")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar=f"{message['agent'][0].upper()}"):
                st.markdown(message["content"])

# Footer
st.markdown("---")
st.markdown(f"A2A Data Analysis System © {pd.Timestamp.now().year} | Powered by Google Gemini Pro")

# Clean up temporary files when the app is done
def cleanup():
    if st.session_state.file_path and os.path.exists(st.session_state.file_path):
        os.unlink(st.session_state.file_path)

# Register the cleanup function
import atexit
atexit.register(cleanup) 