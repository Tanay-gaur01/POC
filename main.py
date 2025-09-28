import os
import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from typing import Optional, List, Any

# Google Genai imports
from google import genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SQLOutputParser(BaseOutputParser):
    """Custom parser to extract SQL query from LLM response."""
    
    def parse(self, text: str) -> str:
        """Parse the LLM output to extract clean SQL query."""
        # Remove any markdown formatting
        text = text.strip()
        
        # Remove ```sql and ``` markers if present
        if text.startswith("```sql"):
            text = text.replace("```sql", "").replace("```", "").strip()
        elif text.startswith("```"):
            text = text.replace("```", "").strip()
        
        # Remove any explanatory text before/after the query
        lines = text.split('\n')
        sql_lines = []
        
        # Flag to track if we've found the start of SQL
        sql_started = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('--') or line.startswith('#'):
                continue
            
            # Stop at explanatory text
            if any(phrase in line.lower() for phrase in ['explanation:', 'note:', 'this query']):
                break
            
            # Check if line starts with SQL keywords or looks like SQL
            if (line.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER')) 
                or sql_started):
                sql_started = True
                sql_lines.append(line)
            # Handle cases where there might be garbage text before SQL
            elif 'SELECT' in line.upper() or 'FROM' in line.upper():
                # Extract SQL part from the line
                sql_start = max(line.upper().find('SELECT'), line.upper().find('WITH'))
                if sql_start >= 0:
                    cleaned_line = line[sql_start:]
                    sql_lines.append(cleaned_line)
                    sql_started = True
        
        sql_query = '\n'.join(sql_lines).strip()
        
        # Additional cleaning for common issues
        # Remove any leading garbage text before SELECT
        if sql_query and not sql_query.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
            select_pos = sql_query.upper().find('SELECT')
            with_pos = sql_query.upper().find('WITH')
            
            start_pos = -1
            if select_pos >= 0 and with_pos >= 0:
                start_pos = min(select_pos, with_pos)
            elif select_pos >= 0:
                start_pos = select_pos
            elif with_pos >= 0:
                start_pos = with_pos
            
            if start_pos >= 0:
                sql_query = sql_query[start_pos:].strip()
        
        # Ensure query ends with semicolon
        if sql_query and not sql_query.endswith(';'):
            sql_query += ';'
        
        # Final validation - if still doesn't look like SQL, return a default query
        if not sql_query or not any(keyword in sql_query.upper() for keyword in ['SELECT', 'FROM']):
            sql_query = "SELECT 'Error: Could not parse SQL query' as message;"
            
        return sql_query


class DatabaseManager:
    """Manages SQLite database operations and schema."""
    
    def __init__(self, db_path: str = "business_data.db"):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Establish database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
            
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def create_business_schema(self):
        """
        Create a fixed, well-designed schema for business data.
        
        Schema Design Rationale:
        - Separate tables for different business domains (financial vs operational)
        - Normalized structure to avoid data redundancy
        - Consistent naming conventions
        - Appropriate data types for business metrics
        - Indexes on frequently queried columns
        """
        
        schema_sql = """
        -- Financial Performance Table
        CREATE TABLE IF NOT EXISTS financial_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            category TEXT NOT NULL,
            sub_category TEXT,
            period TEXT NOT NULL,
            value REAL,
            currency TEXT DEFAULT 'INR',
            unit TEXT DEFAULT 'Crores'
        );
        
        -- Operational Metrics Table  
        CREATE TABLE IF NOT EXISTS operational_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            port_name TEXT NOT NULL,
            commodity TEXT,
            entity_name TEXT,
            operation_type TEXT,
            period TEXT NOT NULL,
            volume REAL,
            unit TEXT DEFAULT 'MMT'
        );
        
        -- ROCE (Return on Capital Employed) Analysis Table
        CREATE TABLE IF NOT EXISTS roce_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            business_unit TEXT NOT NULL,
            port_name TEXT,
            metric_type TEXT NOT NULL,
            period TEXT NOT NULL,
            value REAL,
            calculation_method TEXT
        );
        
        -- Container Handling Data Table
        CREATE TABLE IF NOT EXISTS container_operations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            port_name TEXT NOT NULL,
            operator_entity TEXT,
            container_type TEXT,
            period TEXT NOT NULL,
            volume REAL,
            vehicle_count INTEGER
        );
        
        -- Performance Indexes for faster queries
        CREATE INDEX IF NOT EXISTS idx_financial_period ON financial_performance(period);
        CREATE INDEX IF NOT EXISTS idx_financial_category ON financial_performance(category);
        CREATE INDEX IF NOT EXISTS idx_operational_port ON operational_metrics(port_name);
        CREATE INDEX IF NOT EXISTS idx_operational_period ON operational_metrics(period);
        CREATE INDEX IF NOT EXISTS idx_roce_period ON roce_analysis(period);
        CREATE INDEX IF NOT EXISTS idx_container_port ON container_operations(port_name);
        """
        
        try:
            cursor = self.conn.cursor()
            cursor.executescript(schema_sql)
            self.conn.commit()
            logger.info("Business database schema created successfully")
        except Exception as e:
            logger.error(f"Schema creation failed: {e}")
            raise
    
    def load_csv_data(self, csv_directory: str = "."):
        """Load business data from CSV files in the specified directory."""
        
        csv_directory_path = Path(csv_directory)
        
        # Validate directory
        if not csv_directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {csv_directory}")
        
        if not csv_directory_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {csv_directory}")
        
        logger.info(f"Loading CSV files from directory: {csv_directory_path.absolute()}")
        
        # Define CSV file patterns and their corresponding processing functions
        csv_mappings = {
            # Pattern matching for flexible file naming - check filename contains these patterns
            'balance': ('financial_performance', self._process_balance_sheet),
            'cash_flow': ('financial_performance', self._process_cash_flow),
            'cashflow': ('financial_performance', self._process_cash_flow),
            'consolidated': ('financial_performance', self._process_pnl),
            'pnl': ('financial_performance', self._process_pnl),
            'profit': ('financial_performance', self._process_pnl),
            'quarterly': ('financial_performance', self._process_quarterly_pnl),
            'roce external': ('roce_analysis', self._process_roce_external),
            'roce internal': ('roce_analysis', self._process_roce_internal),  
            'container': ('container_operations', self._process_containers),
            'roro': ('container_operations', self._process_roro),
            'volume': ('operational_metrics', self._process_volumes)
        }
        
        # Find all CSV files in the directory
        csv_files = list(csv_directory_path.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {csv_directory}")
        
        logger.info(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
        
        processed_files = 0
        
        # Process each CSV file
        for csv_file in csv_files:
            file_name_lower = csv_file.name.lower()
            
            # Find matching processor
            processor_found = False
            
            for pattern, (table_name, processor_func) in csv_mappings.items():
                # Check if pattern matches filename (handle spaces and special characters)
                pattern_normalized = pattern.replace('_', ' ').replace('-', ' ')
                file_normalized = file_name_lower.replace('_', ' ').replace('-', ' ')
                
                if pattern_normalized in file_normalized or pattern.replace(' ', '') in file_name_lower.replace(' ', ''):
                    try:
                        logger.info(f"Processing {csv_file.name} → {table_name} (pattern: {pattern})")
                        
                        df = pd.read_csv(csv_file)
                        processed_df = processor_func(df)
                        
                        # Load into database
                        processed_df.to_sql(table_name, self.conn, if_exists='append', index=False)
                        logger.info(f"✅ Loaded {len(processed_df)} records from {csv_file.name}")
                        
                        processed_files += 1
                        processor_found = True
                        break  # Stop after first pattern match
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to load {csv_file.name}: {e}")
                        continue
            
            # Handle files that don't match any pattern
            if not processor_found:
                logger.warning(f"⚠️ No processor found for {csv_file.name} - skipping")
                logger.info(f"   Available patterns: {', '.join(csv_mappings.keys())}")
        
        if processed_files == 0:
            raise ValueError(f"No CSV files could be processed from {csv_directory}")
        
        logger.info(f"✅ Successfully processed {processed_files} out of {len(csv_files)} CSV files")
    
    def _clean_numeric_value(self, value) -> Optional[float]:
        """Clean numeric values (remove commas, currency symbols)."""
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        
        try:
            cleaned = str(value).replace(',', '').replace('₹', '').replace('$', '').strip()
            return float(cleaned) if cleaned else None
        except (ValueError, TypeError):
            return None
    
    def _process_balance_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process balance sheet data into financial_performance format."""
        processed_data = []
        
        for _, row in df.iterrows():
            processed_data.append({
                'metric_name': row.get('Line Item', ''),
                'category': row.get('Category', 'Balance Sheet'),
                'sub_category': row.get('SubCategory', ''),
                'period': row.get('Period', ''),
                'value': self._clean_numeric_value(row.get('Value', 0)),
                'currency': 'INR',
                'unit': 'Crores'
            })
        
        return pd.DataFrame(processed_data)
    
    def _process_cash_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process cash flow data into financial_performance format."""
        processed_data = []
        
        for _, row in df.iterrows():
            processed_data.append({
                'metric_name': row.get('Item', ''),
                'category': 'Cash Flow',
                'sub_category': row.get('Category', ''),
                'period': row.get('Period', ''),
                'value': float(row.get('Value', 0)) if pd.notna(row.get('Value')) else None,
                'currency': 'INR',
                'unit': 'Crores'
            })
        
        return pd.DataFrame(processed_data)
    
    def _process_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process P&L data into financial_performance format."""
        processed_data = []
        
        for _, row in df.iterrows():
            processed_data.append({
                'metric_name': row.get('Line Item', ''),
                'category': 'Profit & Loss',
                'sub_category': '',
                'period': row.get('Period', ''),
                'value': self._clean_numeric_value(row.get('Value', 0)),
                'currency': 'INR',
                'unit': 'Crores'
            })
        
        return pd.DataFrame(processed_data)
    
    def _process_quarterly_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process quarterly P&L data."""
        processed_data = []
        
        for _, row in df.iterrows():
            processed_data.append({
                'metric_name': row.get('Item', ''),
                'category': 'Quarterly Performance',
                'sub_category': row.get('Category', ''),
                'period': row.get('Period', ''),
                'value': float(row.get('Value', 0)) if pd.notna(row.get('Value')) else None,
                'currency': 'INR',
                'unit': 'Crores'
            })
        
        return pd.DataFrame(processed_data)
    
    def _process_roce_external(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process external ROCE data."""
        processed_data = []
        
        for _, row in df.iterrows():
            processed_data.append({
                'business_unit': 'External',
                'port_name': None,
                'metric_type': row.get('Particular', ''),
                'period': row.get('Period', ''),
                'value': self._clean_numeric_value(row.get('Value', 0)),
                'calculation_method': 'External Analysis'
            })
        
        return pd.DataFrame(processed_data)
    
    def _process_roce_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process internal ROCE data."""
        processed_data = []
        
        for _, row in df.iterrows():
            processed_data.append({
                'business_unit': row.get('Category', 'Internal'),
                'port_name': row.get('Port', ''),
                'metric_type': row.get('Line Item', ''),
                'period': row.get('Period', ''),
                'value': self._clean_numeric_value(row.get('Value', 0)),
                'calculation_method': 'Internal Analysis'
            })
        
        return pd.DataFrame(processed_data)
    
    def _process_containers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process container operations data."""
        processed_data = []
        
        for _, row in df.iterrows():
            processed_data.append({
                'port_name': row.get('Port', ''),
                'operator_entity': row.get('Entity', ''),
                'container_type': row.get('Type', ''),
                'period': row.get('Period', ''),
                'volume': float(row.get('Value', 0)) if pd.notna(row.get('Value')) else None,
                'vehicle_count': None
            })
        
        return pd.DataFrame(processed_data)
    
    def _process_roro(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process RORO (Roll-on/Roll-off) vehicle data."""
        processed_data = []
        
        for _, row in df.iterrows():
            processed_data.append({
                'port_name': row.get('Port', ''),
                'operator_entity': None,
                'container_type': 'RORO',
                'period': row.get('Period', ''),
                'volume': float(row.get('Value', 0)) if pd.notna(row.get('Value')) else None,
                'vehicle_count': int(row.get('Number of Cars', 0)) if pd.notna(row.get('Number of Cars')) else None
            })
        
        return pd.DataFrame(processed_data)
    
    def _process_volumes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process cargo volume data."""
        processed_data = []
        
        for _, row in df.iterrows():
            processed_data.append({
                'port_name': row.get('Port', ''),
                'commodity': row.get('Commodity', ''),
                'entity_name': row.get('Entity', ''),
                'operation_type': row.get('Type', ''),
                'period': row.get('Period', ''),
                'volume': float(row.get('Value', 0)) if pd.notna(row.get('Value')) else None,
                'unit': 'MMT'
            })
        
        return pd.DataFrame(processed_data)
    
    def execute_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql_query)
            columns = [description[0] for description in cursor.description]
            results = cursor.fetchall()
            
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_schema_info(self) -> str:
        """Get database schema information for LLM context."""
        return """
DATABASE SCHEMA INFORMATION:

Table 1: financial_performance
- Contains all financial metrics (Revenue, Profit, Assets, Liabilities, etc.)
- Columns: metric_name, category, sub_category, period, value, currency, unit
- Categories: 'Balance Sheet', 'Cash Flow', 'Profit & Loss', 'Quarterly Performance'
- Values in INR Crores

Table 2: operational_metrics  
- Contains port operations and cargo volume data
- Columns: port_name, commodity, entity_name, operation_type, period, volume, unit
- Volume in MMT (Million Metric Tons)
- Covers different ports and commodities

Table 3: roce_analysis
- Contains Return on Capital Employed metrics
- Columns: business_unit, port_name, metric_type, period, value, calculation_method
- Includes both internal and external ROCE analysis

Table 4: container_operations
- Contains container handling and vehicle cargo data
- Columns: port_name, operator_entity, container_type, period, volume, vehicle_count
- Includes RORO (Roll-on/Roll-off) vehicle data

IMPORTANT NOTES:
- All financial values are in INR Crores
- All volume data is in MMT (Million Metric Tons)
- Periods are in format '2024-25', '2023-24'
- Port names: Mundra, APSEZ, etc.
- Use exact column names in queries
"""


class Gemini25FlashClient:
    """Direct client for Gemini 2.5 Flash using Google Genai SDK."""
    
    def __init__(self, google_api_key: str):
        """Initialize Gemini 2.5 Flash client."""
        os.environ['GOOGLE_API_KEY'] = google_api_key
        self.client = genai.Client()
        self.model_name = "gemini-2.5-flash"
        logger.info(f"Initialized Gemini 2.5 Flash client with model: {self.model_name}")
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Gemini 2.5 Flash with retry logic."""
        import time
        
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                
                # Clean the response text to remove common artifacts
                response_text = response.text.strip()
                
                # Remove common prefixes that Gemini sometimes adds
                prefixes_to_remove = ['ite\n', 'ite ', 'sql\n', 'sql ', '```sql\n', '```\n']
                
                for prefix in prefixes_to_remove:
                    if response_text.startswith(prefix):
                        response_text = response_text[len(prefix):].strip()
                        break
                
                return response_text
                
            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"API unavailable, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error("Gemini API is temporarily unavailable. Please try again later.")
                        raise Exception("Gemini API is temporarily unavailable. Please try again in a few moments.")
                else:
                    logger.error(f"Gemini 2.5 Flash API call failed: {e}")
                    raise


class GeminiSQLGenerator:
    """Generates SQL queries using Google's Gemini 2.5 Flash via LangChain."""
    
    def __init__(self, google_api_key: str):
        """Initialize Gemini 2.5 Flash with LangChain."""
        try:
            # Initialize Gemini 2.5 Flash client
            self.gemini_client = Gemini25FlashClient(google_api_key)
            
            # Create the prompt template
            self.prompt_template = self._create_sql_prompt_template()
            
            # Create SQL output parser
            self.output_parser = SQLOutputParser()
            
            logger.info("Gemini 2.5 Flash SQL Generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini 2.5 Flash: {e}")
            raise
    
    def _create_sql_prompt_template(self) -> PromptTemplate:
        """Create a comprehensive prompt template for SQL generation."""
        
        prompt_text = """
You are an expert SQL query generator for a business database. Your task is to convert natural language questions into precise SQLite queries.

CONTEXT:
You are analyzing business data for a port and logistics company with financial and operational metrics.

DATABASE SCHEMA:
{schema_info}

RULES FOR SQL GENERATION:
1. Generate ONLY valid SQLite syntax
2. Use exact table and column names from the schema
3. Always use proper JOINs when combining tables
4. For financial data, assume values are in INR Crores
5. For operational data, assume volumes are in MMT
6. Use appropriate WHERE clauses for filtering
7. Use GROUP BY for aggregations
8. Use ORDER BY to show most relevant results first
9. Limit results to reasonable numbers (use LIMIT clause)
10. Handle NULL values appropriately

QUESTION ANALYSIS GUIDELINES:
- "Revenue" → Look in financial_performance where category = 'Profit & Loss'
- "Profit/Loss" → Look in financial_performance for relevant metric_names
- "Assets/Liabilities" → Look in financial_performance where category = 'Balance Sheet'
- "Cash Flow" → Look in financial_performance where category = 'Cash Flow'
- "Port operations" → Look in operational_metrics
- "Container operations" → Look in container_operations
- "ROCE" → Look in roce_analysis
- "Compare periods" → Use GROUP BY period and ORDER BY
- "Latest/Recent" → Use ORDER BY period DESC

EXAMPLES:
Q: "What is the total revenue for 2024-25?"
A: SELECT SUM(value) as total_revenue FROM financial_performance WHERE metric_name LIKE '%Revenue%' AND period = '2024-25';

Q: "Show me port-wise cargo volumes"
A: SELECT port_name, SUM(volume) as total_volume FROM operational_metrics GROUP BY port_name ORDER BY total_volume DESC;

Q: "Compare assets vs liabilities in the latest period"
A: SELECT category, metric_name, value, period FROM financial_performance WHERE category = 'Balance Sheet' ORDER BY period DESC, value DESC;

HUMAN QUESTION: {question}

Generate a precise SQL query that answers this question. Return ONLY the SQL query without any explanation.

SQL QUERY:
"""
        
        return PromptTemplate(
            input_variables=["schema_info", "question"],
            template=prompt_text
        )
    
    def generate_sql(self, question: str, schema_info: str) -> str:
        """Generate SQL query from natural language question."""
        try:
            logger.info(f"Generating SQL for question: {question}")
            
            # Format the prompt using LangChain template
            formatted_prompt = self.prompt_template.format(
                question=question,
                schema_info=schema_info
            )
            
            # Generate SQL using Gemini 2.5 Flash
            raw_response = self.gemini_client.generate_response(formatted_prompt)
            
            # Parse the response to extract clean SQL
            sql_query = self.output_parser.parse(raw_response)
            
            logger.info(f"Generated SQL: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            raise


class TextToSQLPipeline:
    """Main pipeline that orchestrates the Text-to-SQL process."""
    
    def __init__(self, google_api_key: str, db_path: str = "business_data.db"):
        """Initialize the complete pipeline."""
        self.db_manager = DatabaseManager(db_path)
        self.sql_generator = GeminiSQLGenerator(google_api_key)
        
    def initialize_database(self, csv_directory: str = "."):
        """Initialize database with business data."""
        logger.info("Initializing database...")
        
        # Connect to database
        self.db_manager.connect()
        
        # Create business schema
        self.db_manager.create_business_schema()
        
        # Load CSV data
        self.db_manager.load_csv_data(csv_directory)
        
        logger.info("Database initialization completed")
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Main pipeline: Natural Language → SQL → Results
        
        Args:
            question (str): Natural language question
            
        Returns:
            Dict containing question, SQL query, results, and status
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Step 1: Get database schema information
            schema_info = self.db_manager.get_schema_info()
            
            # Step 2: Generate SQL using Gemini LLM
            sql_query = self.sql_generator.generate_sql(question, schema_info)
            
            # Step 3: Execute SQL query
            results = self.db_manager.execute_query(sql_query)
            
            # Step 4: Return structured response
            response = {
                'question': question,
                'sql_query': sql_query,
                'results': results,
                'results_count': len(results),
                'status': 'success'
            }
            
            logger.info(f"Question processed successfully. Found {len(results)} results.")
            return response
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {
                'question': question,
                'sql_query': None,
                'results': [],
                'results_count': 0,
                'status': 'error',
                'error_message': str(e)
            }
    
    def cleanup(self):
        """Clean up resources."""
        self.db_manager.disconnect()


def main():
    """Main function to demonstrate the Text-to-SQL pipeline."""
    
    import sys
    
    # Get CSV directory from command line argument
    if len(sys.argv) > 1:
        csv_directory = sys.argv[1]
        print(f"📁 Using CSV directory: {os.path.abspath(csv_directory)}")
    else:
        csv_directory = "."
        print("📁 Using current directory for CSV files")
        print("💡 Usage: python main.py [path_to_csv_folder]")
    
    # Check for Google API key
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key:
        print("❌ ERROR: GOOGLE_API_KEY environment variable not found!")
        print("Please set your Google API key in the .env file:")
        print("GOOGLE_API_KEY=your_api_key_here")
        return
    
    # Initialize pipeline
    try:
        pipeline = TextToSQLPipeline(google_api_key)
        
        # Initialize database with CSV data from specified directory
        print(f"🔧 Initializing database from: {csv_directory}")
        pipeline.initialize_database(csv_directory)
        
        print("\n✅ Text-to-SQL Pipeline with Gemini 2.5 Flash Ready!")
        print("\nYou can now ask natural language questions about your business data.")
        print("Examples:")
        print("- What is the total revenue for 2024-25?")
        print("- Show me port-wise cargo volumes")
        print("- Compare assets vs liabilities")
        print("- What are the ROCE metrics for the latest period?")
        
        # Interactive loop
        print("\n" + "="*60)
        print("GEMINI 2.5 FLASH POWERED QUERY INTERFACE")
        print("="*60)
        print("Enter your questions (type 'exit' to quit):")
        
        while True:
            user_question = input("\n❓ Your Question: ").strip()
            
            if user_question.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_question:
                continue
            
            # Process question through pipeline
            print("\n🧠 Thinking with Gemini 2.5 Flash...")
            response = pipeline.process_question(user_question)
            
            # Display results
            if response['status'] == 'success':
                print(f"\n📋 Generated SQL Query:")
                print(f"```sql\n{response['sql_query']}\n```")
                print(f"\n📊 Results ({response['results_count']} records):")
                
                if response['results']:
                    # Display first few results in a readable format
                    for i, result in enumerate(response['results'][:5], 1):
                        print(f"{i}. {dict(result)}")
                    
                    if len(response['results']) > 5:
                        print(f"... and {len(response['results']) - 5} more records")
                else:
                    print("No results found for your query.")
            else:
                print(f"\n❌ Error: {response['error_message']}")
    
    except KeyboardInterrupt:
        print("\n\n👋 System interrupted. Goodbye!")
    
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"\n❌ System error: {e}")
    
    finally:
        try:
            pipeline.cleanup()
        except:
            pass


if __name__ == "__main__":
    main()
