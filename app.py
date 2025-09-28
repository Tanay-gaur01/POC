import streamlit as st
import pandas as pd
import sqlite3
import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our existing classes (assuming main.py is in the same directory)
try:
    from main import (
        DatabaseManager, 
        GeminiSQLGenerator, 
        TextToSQLPipeline,
        SQLOutputParser
    )
except ImportError:
    st.error("❌ Could not import required modules. Make sure main.py is in the same directory.")
    st.stop()

# Check for API key at startup
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in environment variables!")
    st.error("Please create a .env file with: GOOGLE_API_KEY=your_api_key_here")
    st.info("Get your free API key from: https://ai.google.dev/")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Text-to-SQL with Gemini 2.5 Flash",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .sql-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'uploaded_files_info' not in st.session_state:
        st.session_state.uploaded_files_info = []
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = None

def create_temp_directory():
    """Create a temporary directory for uploaded files."""
    if st.session_state.temp_dir is None:
        st.session_state.temp_dir = tempfile.mkdtemp()
    return st.session_state.temp_dir

def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory."""
    temp_dir = create_temp_directory()
    saved_files = []
    
    for uploaded_file in uploaded_files:
        # Save file to temporary directory
        file_path = Path(temp_dir) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(str(file_path))
        
        # Store file info
        file_info = {
            'name': uploaded_file.name,
            'size': uploaded_file.size,
            'type': uploaded_file.type,
            'path': str(file_path)
        }
        
        # Add to session state if not already there
        if file_info not in st.session_state.uploaded_files_info:
            st.session_state.uploaded_files_info.append(file_info)
    
    return saved_files

def initialize_pipeline(csv_directory):
    """Initialize the Text-to-SQL pipeline using API key from environment."""
    try:
        with st.spinner("🔧 Initializing Gemini 2.5 Flash pipeline..."):
            pipeline = TextToSQLPipeline(GOOGLE_API_KEY)
            pipeline.initialize_database(csv_directory)
            return pipeline
    except Exception as e:
        st.error(f"❌ Failed to initialize pipeline: {str(e)}")
        return None

def display_file_preview(file_path):
    """Display a preview of the uploaded CSV file."""
    try:
        df = pd.read_csv(file_path)
        st.write(f"**Preview of {Path(file_path).name}** ({len(df)} rows, {len(df.columns)} columns)")
        st.dataframe(df.head(10), width="stretch")
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def create_natural_language_response(results, question, sql_query):
    """Create a natural language response from query results."""
    if not results:
        return "I couldn't find any data matching your question. The database might not contain the specific information you're looking for."
    
    question_lower = question.lower()
    
    # Single result responses
    if len(results) == 1:
        result = results[0]
        
        # Handle specific question types
        if 'highest' in question_lower and 'volume' in question_lower:
            port = result.get('port_name', 'Unknown port')
            volume = result.get('total_cargo_volume', result.get('total_volume', 0))
            return f"The port with the highest cargo volume is **{port}** with **{volume:,.2f} MMT**."
        
        elif 'revenue' in question_lower:
            revenue = result.get('total_revenue', result.get('value', 0))
            period = result.get('period', '')
            if period:
                return f"The total revenue for {period} is **₹{revenue:,.2f} crores**."
            else:
                return f"The total revenue is **₹{revenue:,.2f} crores**."
        
        elif 'profit' in question_lower or 'loss' in question_lower:
            value = result.get('value', result.get('total', 0))
            metric = result.get('metric_name', 'profit/loss')
            return f"The {metric.lower()} shows **₹{value:,.2f} crores**."
    
    # Multiple results responses
    elif len(results) <= 5:
        if 'port' in question_lower and 'volume' in question_lower:
            port_info = []
            for result in results:
                port = result.get('port_name', result.get('port', 'Unknown'))
                volume = result.get('total_cargo_volume', result.get('total_volume', result.get('volume', 0)))
                port_info.append(f"**{port}**: {volume:,.2f} MMT")
            return f"Port-wise cargo volumes:\n" + "\n".join(port_info)
        
        elif 'revenue' in question_lower or 'financial' in question_lower:
            financial_info = []
            for result in results:
                period = result.get('period', 'Unknown period')
                value = result.get('value', result.get('total_revenue', 0))
                metric = result.get('metric_name', 'Revenue')
                financial_info.append(f"**{period}**: {metric} - ₹{value:,.2f} crores")
            return "Financial performance:\n" + "\n".join(financial_info)
        
        elif 'commodity' in question_lower or 'top' in question_lower:
            commodity_info = []
            for result in results:
                name = result.get('commodity', result.get('metric_name', result.get('port_name', 'Unknown')))
                value = result.get('total_volume', result.get('value', result.get('total', 0)))
                commodity_info.append(f"**{name}**: {value:,.2f}")
            return "Top results:\n" + "\n".join(commodity_info)
    
    # Large result sets
    else:
        result_count = len(results)
        if 'financial' in question_lower or 'revenue' in question_lower or 'profit' in question_lower:
            return f"I found **{result_count} financial records** matching your query. The data includes various financial metrics across different periods and categories."
        elif 'port' in question_lower or 'volume' in question_lower or 'cargo' in question_lower:
            return f"I found **{result_count} operational records** showing cargo volumes and port operations across different time periods."
        else:
            return f"I found **{result_count} records** matching your query with various business metrics and data points."
    
    # Fallback response
    return f"I found **{len(results)} records** matching your query. The data shows various business metrics and operational information."

def main():
    """Main Streamlit application."""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Text-to-SQL with Gemini 2.5 Flash</h1>', unsafe_allow_html=True)
    st.markdown("### Upload your CSV files and ask questions in natural language!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key status
        st.success("🔑 API Key: Loaded from .env file")
        st.info(f"🤖 Model: Gemini 2.5 Flash")
        
        st.markdown("---")
        
        # File upload section
        st.header("📁 Upload CSV Files")
        uploaded_files = st.file_uploader(
            "Choose CSV files",
            accept_multiple_files=True,
            type=['csv'],
            help="Upload multiple CSV files containing your business data"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded!")
            
            # Save uploaded files
            saved_files = save_uploaded_files(uploaded_files)
            
            # Process data button
            if st.button("🚀 Process Data", type="primary"):
                temp_dir = create_temp_directory()
                pipeline = initialize_pipeline(temp_dir)
                
                if pipeline:
                    st.session_state.pipeline = pipeline
                    st.session_state.data_loaded = True
                    st.success("✅ Data processed successfully!")
                    st.rerun()
        
        # Display uploaded files info
        if st.session_state.uploaded_files_info:
            st.markdown("---")
            st.header("📋 Uploaded Files")
            for file_info in st.session_state.uploaded_files_info:
                with st.expander(f"📄 {file_info['name']}"):
                    st.write(f"**Size:** {file_info['size']:,} bytes")
                    st.write(f"**Type:** {file_info['type']}")
    
    # Main content area
    if not st.session_state.data_loaded:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🎯 How it works:")
            st.markdown("""
            1. **📁 Upload your CSV files** (Balance Sheet, P&L, Volumes, etc.)
            2. **🚀 Process the data** to create your database
            3. **❓ Ask questions** in natural language
            4. **📊 Get insights** with SQL queries and visualizations
            
            *API key is automatically loaded from your .env file*
            """)
            
            st.markdown("### 📋 Supported File Types:")
            st.markdown("""
            - **Financial Data**: Balance sheets, P&L statements, cash flow
            - **Operational Data**: Port volumes, container operations
            - **Performance Data**: ROCE analysis, quarterly metrics
            """)
            
        if uploaded_files and not st.session_state.data_loaded:
            st.info("👆 Click 'Process Data' in the sidebar to get started!")
    
    else:
        # Query interface
        st.markdown('<h2 class="sub-header">💬 Ask Your Questions</h2>', unsafe_allow_html=True)
        
        # Example questions
        with st.expander("💡 Example Questions", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Financial Questions:**")
                st.markdown("""
                - What is the total revenue for 2024-25?
                - Show me profit and loss trends
                - Compare assets vs liabilities
                - What's the cash flow for the latest period?
                """)
            
            with col2:
                st.markdown("**Operational Questions:**")
                st.markdown("""
                - Which port has the highest cargo volume?
                - Show container operations by port
                - What are the top commodities by volume?
                - Compare port performance over time
                """)
        
        # Query input
        user_question = st.text_input(
            "❓ Your Question:",
            placeholder="e.g., What is the total revenue for 2024-25?",
            key="query_input"
        )
        
        # Query buttons
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            query_button = st.button("🧠 Ask Gemini", type="primary")
        
        with col2:
            clear_history = st.button("🗑️ Clear History")
            
        if clear_history:
            st.session_state.query_history = []
            st.rerun()
        
        # Process query
        if query_button and user_question and st.session_state.pipeline:
            with st.spinner("🤔 Gemini 2.5 Flash is thinking..."):
                response = st.session_state.pipeline.process_question(user_question)
                
                # Add to history
                st.session_state.query_history.append({
                    'timestamp': datetime.now(),
                    'question': user_question,
                    'response': response
                })
                
                # Display results
                if response['status'] == 'success':
                    # Success header
                    st.markdown('<div class="success-box">✅ <strong>Query Successful!</strong></div>', unsafe_allow_html=True)
                    
                    # Display SQL query
                    st.markdown("**🔍 Generated SQL Query:**")
                    st.code(response['sql_query'], language='sql')
                    
                    # Display results
                    st.markdown(f"**📊 Results ({response['results_count']} records):**")
                    
                    if response['results']:
                        # Convert to DataFrame for better display
                        df_results = pd.DataFrame(response['results'])
                        st.dataframe(df_results, width="stretch")
                        
                        # Generate natural language response
                        nl_response = create_natural_language_response(
                            response['results'], 
                            user_question, 
                            response['sql_query']
                        )
                        
                        st.markdown("**🤖 Natural Language Response:**")
                        st.markdown(nl_response)
                        
                        # Download option
                        csv_data = df_results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results as CSV",
                            csv_data,
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No results found for your query.")
                        st.markdown("**🤖 Natural Language Response:**")
                        st.markdown("I couldn't find any data matching your question. Please try rephrasing your query or check if the information exists in the uploaded data.")
                        
                else:
                    # Error display
                    st.markdown(f'<div class="error-box">❌ <strong>Error:</strong> {response["error_message"]}</div>', unsafe_allow_html=True)
        
        # Query History
        if st.session_state.query_history:
            st.markdown("---")
            st.markdown('<h2 class="sub-header">📜 Query History</h2>', unsafe_allow_html=True)
            
            for i, entry in enumerate(reversed(st.session_state.query_history[-10:])):  # Show last 10
                with st.expander(f"Q{len(st.session_state.query_history)-i}: {entry['question'][:50]}..."):
                    st.markdown(f"**🕒 Time:** {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown(f"**❓ Question:** {entry['question']}")
                    st.markdown(f"**🔍 SQL:** `{entry['response']['sql_query']}`")
                    st.markdown(f"**📊 Results:** {entry['response']['results_count']} records")
                    
                    if entry['response']['status'] == 'success' and entry['response']['results']:
                        df_hist = pd.DataFrame(entry['response']['results'])
                        st.dataframe(df_hist.head(3), width="stretch")
    
    # Data Preview Section
    if st.session_state.uploaded_files_info:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📋 Data Preview</h2>', unsafe_allow_html=True)
        
        # File selector for preview
        file_names = [info['name'] for info in st.session_state.uploaded_files_info]
        selected_file = st.selectbox("Select file to preview:", file_names)
        
        if selected_file:
            # Find the corresponding file info
            selected_info = next(info for info in st.session_state.uploaded_files_info if info['name'] == selected_file)
            display_file_preview(selected_info['path'])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🧠 <strong>Powered by Gemini 2.5 Flash</strong> | 
        🔗 Built with Streamlit | 
        🚀 Text-to-SQL Magic
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
