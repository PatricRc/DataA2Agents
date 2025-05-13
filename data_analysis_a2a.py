#!/usr/bin/env python3
"""
A2A Data Analysis System - Main Entry Point

This system provides a set of specialized data agents that can work together:
- Data Analyst: Analyzes data and provides statistical insights
- Data Scientist: Builds and evaluates machine learning models
- Data Visualization Analyst: Creates visualizations and charts
- Data Storyteller: Creates narratives from data insights

The system allows users to upload CSV/Excel files and interact with the agents via command line interface.
Uses Google's Gemini Pro model for AI capabilities.
"""

import os
import sys
import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import pandas as pd
import numpy as np
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from dotenv import load_dotenv
import tkinter as tk
from tkinter import filedialog

# Configure Gemini API if available
if genai.is_available():
    # Load environment variables from .env file
    load_dotenv()

    # Get Gemini API key from environment
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY environment variable not set")
        print("Please create a .env file with your GEMINI_API_KEY=your_key or provide it in the app")
    else:
        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)

        # Get Gemini model
        model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config={
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 2048,
            },
        )
else:
    GEMINI_API_KEY = None
    model = None

# Global variables
current_df = None
current_file_path = None
chat_sessions = {}  # Store chat sessions for each agent


class BaseAgent:
    """Base class for all data analysis agents."""
    
    def __init__(self):
        self.name = "Base Agent"
        self.description = "Base agent for data analysis."
        
    def process(self, user_input: str, df: pd.DataFrame, api_key: str) -> str:
        """
        Process the user input and return a response.
        
        Args:
            user_input: The user's query or request
            df: The pandas DataFrame containing the data
            api_key: The API key for Gemini
            
        Returns:
            A string response to the user's query
        """
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Get DataFrame info as context
        df_info = self._get_dataframe_info(df)
        
        # Process with the model
        response = self._call_model(user_input, df_info)
        
        return response
    
    def _get_dataframe_info(self, df: pd.DataFrame) -> str:
        """Get information about the DataFrame to use as context."""
        buffer = io.StringIO()
        
        # Basic info
        buffer.write(f"DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")
        
        # Column information
        buffer.write("Column information:\n")
        for col in df.columns:
            buffer.write(f"- {col} ({df[col].dtype}): ")
            
            # Add some sample values
            sample_values = df[col].dropna().sample(min(3, df[col].count())).tolist()
            buffer.write(f"Sample values: {sample_values}\n")
            
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            buffer.write("\nSummary statistics for numeric columns:\n")
            buffer.write(df[numeric_cols].describe().to_string())
            
        return buffer.getvalue()
    
    def _call_model(self, user_input: str, context: str) -> str:
        """Call the Gemini model with the user's query and context."""
        # Create the prompt
        prompt = f"""You are the {self.name}, {self.description}
        
Here is information about the data:
{context}

User query: {user_input}

Provide a helpful, accurate, and concise response based on the data information provided.
"""
        
        # Call the model
        model = genai.GenerativeModel(model_name="gemini-pro")
        response = model.generate_content(prompt)
        
        return response.text


class DataAnalyst(BaseAgent):
    """Agent for general data analysis and statistical insights."""
    
    def __init__(self):
        super().__init__()
        self.name = "Data Analyst"
        self.description = "I analyze data to provide statistical insights and answer questions about patterns and trends."
    
    def process(self, user_input: str, df: pd.DataFrame, api_key: str) -> str:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Get DataFrame info as context
        df_info = self._get_dataframe_info(df)
        
        # Create a more specific prompt for this agent
        prompt = f"""You are a Data Analyst focused on exploring data and providing statistical insights.

Here is information about the data:
{df_info}

User query: {user_input}

Analyze the data and provide statistical insights. Include:
1. Relevant descriptive statistics
2. Key observations about distributions, trends, or patterns
3. Possible correlations or relationships in the data
4. Answers to specific questions asked by the user

Your response should be clear, concise, and focused on the data.
"""
        
        # Call the model
        model = genai.GenerativeModel(model_name="gemini-pro")
        response = model.generate_content(prompt)
        
        return response.text


class DataScientist(BaseAgent):
    """Agent for predictive modeling and machine learning tasks."""
    
    def __init__(self):
        super().__init__()
        self.name = "Data Scientist"
        self.description = "I build and evaluate machine learning models to make predictions and identify patterns."
    
    def process(self, user_input: str, df: pd.DataFrame, api_key: str) -> str:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Get DataFrame info as context
        df_info = self._get_dataframe_info(df)
        
        # Create a more specific prompt for this agent
        prompt = f"""You are a Data Scientist focused on predictive modeling and machine learning.

Here is information about the data:
{df_info}

User query: {user_input}

Consider this a machine learning consultation. Provide:
1. An assessment of what ML tasks are appropriate for this data
2. Recommendations for feature engineering or preprocessing steps
3. Suggestions for appropriate models and evaluation metrics
4. Potential challenges and limitations
5. A high-level approach to solve the user's specific problem

Your response should be clear, practical, and focused on applying ML to this data.
"""
        
        # Call the model
        model = genai.GenerativeModel(model_name="gemini-pro")
        response = model.generate_content(prompt)
        
        return response.text


class DataVisualizationAnalyst(BaseAgent):
    """Agent for creating data visualizations and charts."""
    
    def __init__(self):
        super().__init__()
        self.name = "Data Visualization Analyst"
        self.description = "I create visualizations and charts to represent data in meaningful ways."
    
    def process(self, user_input: str, df: pd.DataFrame, api_key: str) -> str:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Get DataFrame info as context
        df_info = self._get_dataframe_info(df)
        
        # Create a more specific prompt for this agent
        prompt = f"""You are a Data Visualization Analyst focused on creating effective visual representations of data.

Here is information about the data:
{df_info}

User query: {user_input}

Provide visualization recommendations:
1. Suggest the most appropriate chart types for this data
2. Explain what insights each visualization would reveal
3. Recommend color schemes and design considerations
4. Provide code examples using matplotlib, seaborn, or plotly if requested
5. Explain best practices for visualizing this specific data

Your response should focus on effective visualization choices and techniques.
"""
        
        # Call the model
        model = genai.GenerativeModel(model_name="gemini-pro")
        response = model.generate_content(prompt)
        
        return response.text


class DataStoryteller(BaseAgent):
    """Agent for creating narratives from data insights."""
    
    def __init__(self):
        super().__init__()
        self.name = "Data Storyteller"
        self.description = "I create narratives and stories from data insights to communicate findings effectively."
    
    def process(self, user_input: str, df: pd.DataFrame, api_key: str) -> str:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Get DataFrame info as context
        df_info = self._get_dataframe_info(df)
        
        # Create a more specific prompt for this agent
        prompt = f"""You are a Data Storyteller focused on creating compelling narratives from data insights.

Here is information about the data:
{df_info}

User query: {user_input}

Create a data story that:
1. Identifies the most interesting or important insights in this data
2. Weaves these insights into a cohesive narrative
3. Highlights key points that would resonate with stakeholders
4. Suggests how this story could be presented (executive summary, presentation, blog post, etc.)
5. Provides a clear and memorable message supported by the data

Your response should be engaging, clear, and focused on communicating insights effectively.
"""
        
        # Call the model
        model = genai.GenerativeModel(model_name="gemini-pro")
        response = model.generate_content(prompt)
        
        return response.text


class DataManager:
    """Manages data loading and basic operations."""
    
    def __init__(self):
        self.data = None
        self.file_path = None
        
    def load_data(self, file_path):
        """Load data from a CSV or Excel file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if file_path.suffix.lower() == '.csv':
            self.data = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            self.data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
        self.file_path = file_path
        return self.data
        
    def get_info(self):
        """Get information about the dataset."""
        if self.data is None:
            return "No data loaded."
            
        info = {
            "filename": self.file_path.name,
            "shape": self.data.shape,
            "columns": [
                {
                    "name": col,
                    "dtype": str(self.data[col].dtype),
                    "unique_values": self.data[col].nunique(),
                    "missing_values": self.data[col].isna().sum()
                }
                for col in self.data.columns
            ]
        }
        return info
        
    def head(self, n=5):
        """Get the first n rows of the dataset."""
        if self.data is None:
            return "No data loaded."
        return self.data.head(n)
        
    def tail(self, n=5):
        """Get the last n rows of the dataset."""
        if self.data is None:
            return "No data loaded."
        return self.data.tail(n)
        
    def describe(self):
        """Get statistical description of the dataset."""
        if self.data is None:
            return "No data loaded."
        return self.data.describe()


class CommandLineInterface:
    """Command-line interface for the A2A Data Analysis System."""
    
    def __init__(self, data_manager, agents):
        self.data_manager = data_manager
        self.agents = agents
        
    def do_load(self, arg=None):
        """Load a CSV or Excel file."""
        if not arg:
            # Open file dialog if no path is provided
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            file_path = filedialog.askopenfilename(
                title="Select a CSV or Excel file",
                filetypes=[
                    ("Data files", "*.csv;*.xlsx;*.xls"),
                    ("CSV files", "*.csv"),
                    ("Excel files", "*.xlsx;*.xls"),
                    ("All files", "*.*")
                ]
            )
            root.destroy()
            
            if not file_path:  # User cancelled the dialog
                print("File selection cancelled.")
                return
            
            arg = file_path
            
        # Load the file
        try:
            print(f"Loading file: {arg}")
            self.data_manager.load_data(arg)
            print(f"Loaded {self.data_manager.data.shape[0]} rows and {self.data_manager.data.shape[1]} columns.")
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            
    def do_info(self):
        """Display information about the current dataset."""
        if self.data_manager.data is None:
            print("No dataset loaded. Use 'load' to load a dataset.")
            return
            
        info = self.data_manager.get_info()
        print(f"Current dataset: {info['filename']}")
        print(f"Shape: {info['shape'][0]} rows, {info['shape'][1]} columns")
        print("\nColumn information:")
        for col in info["columns"]:
            print(f"  - {col['name']} ({col['dtype']}): {col['unique_values']} unique values, {col['missing_values']} missing")
            
    def do_head(self, n=5):
        """Display the first n rows of the dataset."""
        if self.data_manager.data is None:
            print("No dataset loaded. Use 'load' to load a dataset.")
            return
            
        try:
            if n:
                n = int(n)
        except ValueError:
            print("Invalid number. Using default (5).")
            n = 5
            
        print(self.data_manager.head(n))
        
    def do_tail(self, n=5):
        """Display the last n rows of the dataset."""
        if self.data_manager.data is None:
            print("No dataset loaded. Use 'load' to load a dataset.")
            return
            
        try:
            if n:
                n = int(n)
        except ValueError:
            print("Invalid number. Using default (5).")
            n = 5
            
        print(self.data_manager.tail(n))
        
    def do_describe(self):
        """Display statistical description of the dataset."""
        if self.data_manager.data is None:
            print("No dataset loaded. Use 'load' to load a dataset.")
            return
            
        print(self.data_manager.describe())
        
    def do_help(self):
        """Display help information."""
        print("\nAvailable commands:")
        print("  load [file_path]       - Load a CSV or Excel file (opens file dialog if no path provided)")
        print("  info                   - Show information about the current dataset")
        print("  head [n]               - Show the first n rows (default: 5)")
        print("  tail [n]               - Show the last n rows (default: 5)")
        print("  describe               - Show statistical description of the dataset")
        print("  analyst <query>        - Ask the Data Analyst")
        print("  scientist <query>      - Ask the Data Scientist")
        print("  visualizer <query>     - Ask the Data Visualization Analyst")
        print("  storyteller <query>    - Ask the Data Storyteller")
        print("  clear                  - Clear the screen")
        print("  help                   - Show this help message")
        print("  exit                   - Exit the application")
        
    def do_clear(self):
        """Clear the screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print_header()
        
    def do_agent(self, agent_name, query):
        """Query an agent."""
        if agent_name not in self.agents:
            print(f"Unknown agent: {agent_name}")
            return
            
        if not query:
            print(f"Please provide a query for the {agent_name.title()}")
            return
            
        if self.data_manager.data is None:
            print("No dataset loaded. Use 'load' to load a dataset.")
            return
            
        try:
            print(f"\nProcessing your query with the {self.agents[agent_name].name}...")
            result = self.agents[agent_name].process(
                user_input=query, 
                df=self.data_manager.data,
                api_key=GEMINI_API_KEY
            )
            print(f"\n{self.agents[agent_name].name} Response:")
            print("-" * 80)
            print(result)
            print("-" * 80)
        except Exception as e:
            print(f"Error from {agent_name}: {str(e)}")


def print_header():
    """Print the application header."""
    print("\n" + "=" * 80)
    print(" " * 25 + "A2A DATA ANALYSIS SYSTEM")
    print(" " * 20 + "Powered by Google Gemini Pro")
    print("=" * 80)


def main():
    """Main function to run the A2A data analysis system."""
    print_header()
    print("\nWelcome to the A2A Data Analysis System!")
    print("Type 'help' to see available commands.")
    
    # Initialize data manager and agents
    data_manager = DataManager()
    agents = {
        "analyst": DataAnalyst(),
        "scientist": DataScientist(),
        "visualizer": DataVisualizationAnalyst(),
        "storyteller": DataStoryteller()
    }
    
    # Initialize CLI
    cli = CommandLineInterface(data_manager, agents)
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("\n> ").strip()
            
            # Parse the command
            if not user_input:
                continue
                
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None
            
            # Process commands
            if command == "exit":
                print("Exiting the A2A Data Analysis System. Goodbye!")
                break
                
            elif command == "help":
                cli.do_help()
                
            elif command == "clear":
                cli.do_clear()
                
            elif command == "load":
                cli.do_load(arg)
                
            elif command == "info":
                cli.do_info()
                
            elif command == "head":
                cli.do_head(arg)
                
            elif command == "tail":
                cli.do_tail(arg)
                
            elif command == "describe":
                cli.do_describe()
                
            elif command in ["analyst", "scientist", "visualizer", "storyteller"]:
                cli.do_agent(command, arg)
                
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' to see available commands.")
                
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main() 