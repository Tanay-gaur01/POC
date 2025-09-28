# SQL System POC with Gemini 2.5 Flash

A smart system that lets you ask business questions in plain English and get answers from your data.

## What This Does

This system converts your natural language questions (like "What's our revenue this year?") into SQL database queries, runs them against your business data, and gives you answers in plain English.

**Example:**
- You ask: "Which port has the highest cargo volume?"
- System generates: `SELECT port_name, SUM(volume) FROM operational_metrics GROUP BY port_name ORDER BY SUM(volume) DESC LIMIT 1;`
- You get: "The port with the highest cargo volume is Mundra with 757.27 MMT."

## Quick Start

### Step 1: Install Python Requirements
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Mac/Linux
# or venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

### Step 2: Get Google API Key
1. Go to [Google AI Studio](https://ai.google.dev/)
2. Create a free account and get an API key
3. Create a `.env` file in your project folder:
```
GOOGLE_API_KEY=your_api_key_here
```

### Step 3: Run the System

**Option A: Web Interface (Recommended)**
```bash
streamlit run app.py
```
Then open your browser to the URL shown (usually http://localhost:8502)

**Option B: Command Line**
```bash
python main.py
```

## Dependencies (requirements.txt)

```
pandas>=2.0.0
python-dotenv>=1.0.0
langchain>=0.1.0
google-genai>=0.7.0
streamlit>=1.28.0
```

## Database Schema

The system creates 4 main tables from your CSV files:

### financial_performance
Stores all financial data (revenue, profit, assets, etc.)
- `metric_name` - What is being measured (e.g., "Total Revenue")
- `category` - Type of financial data ("Profit & Loss", "Balance Sheet", etc.)
- `period` - Time period ("2024-25", "2023-24")
- `value` - The actual number
- `currency` - Always "INR"
- `unit` - Always "Crores"

### operational_metrics
Stores port operations and cargo data
- `port_name` - Which port (e.g., "Mundra")
- `commodity` - What was shipped (e.g., "Crude Oil")
- `volume` - How much (in Million Metric Tons)
- `period` - When

### roce_analysis
Stores Return on Capital Employed calculations
- `business_unit` - Internal or External analysis
- `metric_type` - What's being measured (e.g., "EBITDA")
- `value` - The calculated value
- `period` - Time period

### container_operations
Stores container handling data
- `port_name` - Which port
- `container_type` - Type of container operation
- `volume` - Amount handled
- `vehicle_count` - Number of vehicles (for RORO operations)

**Why this design?**
- Each table handles one type of business data
- No duplicate information
- Easy to query and understand
- Matches how businesses actually think about their data

## Design Choices

### Why Gemini 2.5 Flash?
- **Latest and fastest** Google AI model
- **Free tier available** - perfect for proof of concept
- **Great at understanding** business context and generating SQL
- **Reliable and consistent** responses

### Why LangChain?
- **Professional framework** for working with AI models
- **Structured prompts** - we can give the AI clear instructions
- **Easy to maintain** and update
- **Industry standard** for AI applications

### Our Prompting Strategy
We give the AI detailed instructions including:
- **Complete database schema** - so it knows what tables and columns exist
- **Business context** - explains this is port/logistics company data  
- **Clear rules** - only generate valid SQLite, use exact column names, etc.
- **Example queries** - shows the AI what good queries look like
- **Formatting requirements** - return only SQL, no explanations

### Architecture Decisions

**Streamlit Web Interface:**
- Easy for non-technical users
- Upload CSV files through the browser
- No command line needed

**SQLite Database:**
- Simple and reliable
- Perfect for proof of concept
- No separate database server needed

**Modular Code Design:**
- `DatabaseManager` - handles all database operations
- `GeminiSQLGenerator` - converts questions to SQL
- `TextToSQLPipeline` - coordinates everything
- Easy to test and modify individual parts

## How to Use

### Web Interface
1. Run `streamlit run app.py`
2. Upload your CSV files
3. Click "Process Data"
4. Ask questions like:
   - "What is the total revenue for 2024-25?"
   - "Which port handles the most cargo?"
   - "Show me profit and loss trends"
   - "Compare assets vs liabilities"

### What Files to Upload
The system expects business CSV files like:
- Balance sheets
- Profit & Loss statements
- Cash flow data
- Port operational data
- Container handling data
- ROCE analysis

File names should contain keywords like "balance", "pnl", "volume", "containers", etc.

## Limitations and Known Issues

### Current Limitations
1. **Only works with the specific business domain** - designed for port/logistics companies
2. **CSV files must follow expected format** - column names matter
3. **No data validation** - assumes clean, well-formatted CSV files
4. **Limited to SQLite** - not designed for huge datasets
5. **English questions only** - doesn't handle other languages

### Known Issues
1. **Complex multi-table queries** can sometimes be incorrect
2. **Very specific or unusual questions** might not work well
3. **API rate limits** - free Google AI tier has usage limits
4. **File upload size limits** - very large CSV files might cause issues

### Questions That Work Best
- Simple data requests: "What is our revenue?"
- Comparisons: "Compare this year vs last year"
- Top/bottom lists: "Which ports handle the most cargo?"
- Trend analysis: "Show me quarterly performance"

### Questions That Might Not Work
- Complex calculations across multiple time periods
- Questions requiring business logic not in the data
- Personal information (system correctly says it doesn't have this data)
- Questions about data not in your CSV files

## System Requirements

- Python 3.8 or higher
- Internet connection (for Google AI API)
- Modern web browser (for Streamlit interface)
- At least 4GB RAM recommended

## Troubleshooting

**"API key not found"**
- Make sure your `.env` file is in the same folder as the code
- Check that your API key is correct

**"No CSV files found"**
- Make sure your CSV files are uploaded through the web interface
- Check file names contain recognizable keywords

**"SQL generation failed"**
- Try rephrasing your question
- Make sure you're asking about data that exists in your files
- Check that the AI service is available



This system successfully demonstrates how AI can make business data accessible to everyone, not just technical users.
