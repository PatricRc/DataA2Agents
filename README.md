# A2A Data Analysis System

An advanced data analysis system that provides a set of specialized AI agents powered by Google's Gemini 2.0 Flash model. This system follows the Agent-to-Agent (A2A) protocol initiated by Google for AI agent interoperability.

## Features

- **Multiple Specialized Agents**: 
  - **Data Analyst**: Analyzes data and provides statistical insights
  - **Data Scientist**: Builds and evaluates machine learning models
  - **Data Visualization Analyst**: Creates visualizations and charts
  - **Data Storyteller**: Creates narratives from data insights

- **Data Loading**: Upload and analyze CSV or Excel files
- **Interactive Interfaces**:
  - Command-line interface for direct interaction
  - Web API for integration with other applications
  - Streamlit app for user-friendly visual interaction
- **Powered by Google Gemini 2.0 Flash**: Utilizes state-of-the-art AI capabilities

## Installation

1. Clone this repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

You can get a Gemini API key from [Google AI Studio](https://ai.google.dev/).

## Usage

### Command-Line Interface

Run the command-line interface:

```
python data_analysis_a2a.py
```

#### Available commands:

- `load <file_path>` - Load a CSV or Excel file
- `info` - Show information about the current dataset
- `head [n]` - Show the first n rows (default: 5)
- `tail [n]` - Show the last n rows (default: 5)
- `describe` - Show statistical description of the dataset
- `analyst <query>` - Ask the Data Analyst
- `scientist <query>` - Ask the Data Scientist
- `visualizer <query>` - Ask the Data Visualization Analyst
- `storyteller <query>` - Ask the Data Storyteller
- `clear` - Clear the screen
- `help` - Show the help message
- `exit` - Exit the application

### Web Interface

Run the web interface:

```
python web_interface.py
```

This will start a FastAPI server on port 8000. You can access the API documentation at `http://localhost:8000/docs`.

### Streamlit App (New!)

Run the Streamlit app for a user-friendly interface:

#### On Windows:
Using batch file (recommended for most users):
```
run_streamlit.bat
```

Using PowerShell:
```
.\run_streamlit.ps1
```

#### On Linux/macOS:
```
python streamlit_app.py
```

Or use the provided bash script:
```
chmod +x run_streamlit.sh  # Make executable (first time only)
./run_streamlit.sh
```

This will start a Streamlit app on port 8501 and automatically open it in your browser. 
If it doesn't open automatically, visit `http://localhost:8501`.

#### Troubleshooting:
If you encounter errors when launching the Streamlit app:

1. Ensure all dependencies are installed:
   ```
   pip install -r requirements.txt
   ```

2. If you see an error about missing `google.generativeai`:
   ```
   pip install google-generativeai==0.8.5
   ```

3. If you see permission errors with PowerShell scripts:
   ```
   powershell -ExecutionPolicy Bypass -File run_streamlit.ps1
   ```

The Streamlit app provides:
- Easy file upload functionality
- Dataset preview and statistical information
- Interactive chat interface with all agents
- Conversation history tracking

## Example Queries

### Data Analyst
- "Perform a correlation analysis on the numeric columns"
- "Identify outliers in the dataset"
- "What are the key statistical insights from this data?"

### Data Scientist
- "What machine learning models would be appropriate for predicting [column_name]?"
- "How should I handle the missing values in this dataset?"
- "What feature engineering steps would you recommend?"

### Data Visualization Analyst
- "Create a visualization to show the relationship between [column1] and [column2]"
- "What's the best way to visualize the distribution of [column_name]?"
- "Generate code for an interactive dashboard for this data"

### Data Storyteller
- "Create an executive summary of the key insights from this data"
- "Craft a narrative explaining the trends in [column_name] over time"
- "How would you explain these findings to a non-technical audience?"

## System Architecture

The system follows an Agent-to-Agent (A2A) architecture where each specialized agent can:
- Process queries related to their domain expertise
- Generate responses using the Gemini 2.0 Flash model
- Maintain conversation context for follow-up questions

The system provides multiple interfaces:
- Command-line interface for direct interaction
- Web API with RESTful endpoints for application integration
- Streamlit app for interactive visual exploration

## Dependencies

- google-generativeai>=0.8.5
- pandas>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- plotly>=5.13.0
- scikit-learn>=1.3.0
- fastapi>=0.100.0 (for web interface)
- uvicorn>=0.23.0 (for web interface)
- streamlit>=1.31.0 (for Streamlit app)
- python-dotenv>=1.0.0
- openpyxl>=3.1.0 (for Excel support)

## License

MIT

## Acknowledgements

- Google for providing the Gemini API and A2A protocol
- The open-source data science community
- Streamlit for the interactive app framework

## Disclaimer

This application is a demonstration of using generative AI for data analysis and does not guarantee accuracy of the analysis or predictions. Always verify the results with proper validation methods. 