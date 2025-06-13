import streamlit as st
import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import traceback
from datetime import datetime
import pandas as pd
import pyodbc

# Core imports
from langchain_community.utilities import SQLDatabase
from sqlalchemy import URL, create_engine, text
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration dataclass"""
    db_type: str = "SQL Server"
    server: str = ""
    database: str = ""
    username: str = ""
    password: str = ""
    driver: str = "ODBC Driver 17 for SQL Server"
    additional_params: Dict[str, str] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}

@dataclass
class LLMConfig:
    """LLM configuration dataclass"""
    provider: str
    model: str
    api_key: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None

class SQLAgentManager:
    """Production-level SQL Agent Manager"""
    
    SUPPORTED_DB_DRIVERS = {
        "ODBC Driver 17 for SQL Server": "ODBC Driver 17 for SQL Server",
        "ODBC Driver 18 for SQL Server": "ODBC Driver 18 for SQL Server",
        "FreeTDS": "FreeTDS",
    }
    
    SUPPORTED_LLM_PROVIDERS = {
        "OpenAI": ChatOpenAI,
        "Groq": ChatGroq,
        "Google": ChatGoogleGenerativeAI
    }
    
    def __init__(self):
        self.db = None
        self.llm = None
        self.agent_executor = None
        self.connection_url = None
        
    def create_connection_url(self, config: DatabaseConfig) -> URL:
        """Create SQLAlchemy connection URL from config"""
        try:
            if not all([config.server, config.database, config.username, config.password]):
                raise ValueError("All database connection parameters are required")
            
            return URL.create(
                "mssql+pyodbc",
                username=config.username,
                password=config.password,
                host=config.server,
                database=config.database,
                query={"driver": config.driver}
            )
        except Exception as e:
            logger.error(f"Error creating connection URL: {str(e)}")
            raise
    
    def test_connection(self, config: DatabaseConfig) -> tuple[bool, str]:
        """Test database connection"""
        try:
            connection_url = self.create_connection_url(config)
            engine = create_engine(connection_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, "Connection successful!"
        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def initialize_database(self, config: DatabaseConfig) -> bool:
        """Initialize database connection"""
        try:
            self.connection_url = self.create_connection_url(config)
            self.db = SQLDatabase.from_uri(self.connection_url)
            logger.info(f"Database initialized successfully: {config.db_type}")
            return True
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            st.error(f"Database initialization failed: {str(e)}")
            return False
    
    def initialize_llm(self, config: LLMConfig) -> bool:
        """Initialize LLM"""
        try:
            if config.provider == "OpenAI":
                os.environ["OPENAI_API_KEY"] = config.api_key
                self.llm = ChatOpenAI(
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )
            elif config.provider == "Groq":
                os.environ["GROQ_API_KEY"] = config.api_key
                self.llm = ChatGroq(
                    model_name=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )
            elif config.provider == "Google":
                os.environ["GOOGLE_API_KEY"] = config.api_key
                self.llm = ChatGoogleGenerativeAI(
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {config.provider}")
            
            logger.info(f"LLM initialized successfully: {config.provider} - {config.model}")
            return True
        except Exception as e:
            logger.error(f"LLM initialization failed: {str(e)}")
            st.error(f"LLM initialization failed: {str(e)}")
            return False
    
    def create_agent(self) -> bool:
        """Create the SQL agent"""
        try:
            if not self.db or not self.llm:
                raise ValueError("Database and LLM must be initialized first")
            
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            tools = toolkit.get_tools()
            
            system_message = """
You are a professional SQL database analyst designed to interact with SQL databases safely and efficiently.

CORE RESPONSIBILITIES:
- Generate syntactically correct {dialect} queries
- Provide accurate data analysis and insights
- Maintain data security and query safety

MANDATORY SAFETY RULES:
1. NEVER execute DML statements (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE)
2. ALWAYS limit results to maximum {top_k} rows unless specifically requested
3. EXCLUDE sensitive columns (passwords, tokens, keys, personal identifiers)
4. INCLUDE meaningful columns (names, descriptions, titles, categories)
5. Use specific column names in SELECT - NEVER use SELECT *

QUERY PROCESS:
1. ALWAYS examine available tables first
2. Query relevant table schemas 
3. Construct optimized queries with proper JOINs when needed
4. Validate query syntax before execution
5. Format results in clear, readable tables

DATA PRIVACY:
- Deny requests for sensitive personal data
- Exclude ID columns unless specifically relevant
- Include descriptive columns (names, titles) when available
- Respect data classification and access controls

RESPONSE FORMAT:
- Provide clear explanations of your analysis
- Show the SQL query used
- Present results in structured format
- Offer insights and observations when relevant

Remember: You are a data analyst, not a database administrator. Focus on querying and analysis, never on database modifications.
            """.format(
                dialect=self.db.dialect if self.db else "SQL",
                top_k=100
            )
            
            self.agent_executor = create_react_agent(
                self.llm, 
                tools, 
                prompt=system_message
            )
            
            logger.info("SQL Agent created successfully")
            return True
        except Exception as e:
            logger.error(f"Agent creation failed: {str(e)}")
            st.error(f"Agent creation failed: {str(e)}")
            return False
    
    def execute_query(self, question: str) -> Dict[str, Any]:
        """Execute query through the agent"""
        try:
            if not self.agent_executor:
                raise ValueError("Agent not initialized")
            
            logger.info(f"Executing query: {question}")
            
            response_data = {
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "steps": [],
                "final_answer": "",
                "error": None
            }
            
            for step in self.agent_executor.stream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="values",
            ):
                if step.get("messages"):
                    last_message = step["messages"][-1]
                    response_data["steps"].append({
                        "type": last_message.__class__.__name__,
                        "content": last_message.content if hasattr(last_message, 'content') else str(last_message)
                    })
                    response_data["final_answer"] = last_message.content if hasattr(last_message, 'content') else str(last_message)
            
            return response_data
            
        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "steps": [],
                "final_answer": "",
                "error": error_msg
            }
    
    def get_table_info(self) -> Dict[str, Any]:
        """Get database table information"""
        try:
            if not self.db:
                return {"error": "Database not initialized"}
            
            tables = self.db.get_usable_table_names()
            table_info = {}
            for table in tables[:10]:  # Limit to first 10 tables
                try:
                    schema = self.db.get_table_info([table])
                    table_info[table] = schema
                except Exception as e:
                    table_info[table] = f"Error getting schema: {str(e)}"
            
            return {
                "dialect": self.db.dialect,
                "total_tables": len(tables),
                "tables": tables,
                "sample_schemas": table_info
            }
        except Exception as e:
            return {"error": f"Failed to get table info: {str(e)}"}

def create_streamlit_ui():
    """Create the Streamlit UI"""
    st.set_page_config(
        page_title="SQL Database Agent",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Production SQL Database Agent")
    st.markdown("---")
    
    # Initialize session state
    if 'agent_manager' not in st.session_state:
        st.session_state.agent_manager = SQLAgentManager()
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    if 'llm_initialized' not in st.session_state:
        st.session_state.llm_initialized = False
    if 'agent_ready' not in st.session_state:
        st.session_state.agent_ready = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Database Configuration
        st.subheader("📊 Database Settings")
        with st.expander("Database Connection Details", expanded=True):
            server = st.text_input("Server", placeholder="server-name\\instance-name")
            database = st.text_input("Database", placeholder="database-name")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            driver = st.selectbox(
                "Driver",
                options=list(SQLAgentManager.SUPPORTED_DB_DRIVERS.keys()),
                index=0
            )
        
        if st.button("🔗 Test Database Connection", type="secondary"):
            if not all([server, database, username, password]):
                st.error("Please provide all database connection details")
            else:
                db_config = DatabaseConfig(
                    server=server,
                    database=database,
                    username=username,
                    password=password,
                    driver=driver
                )
                with st.spinner("Testing connection..."):
                    success, message = st.session_state.agent_manager.test_connection(db_config)
                    if success:
                        st.success(message)
                        if st.session_state.agent_manager.initialize_database(db_config):
                            st.session_state.db_connected = True
                    else:
                        st.error(message)
        
        st.markdown("---")
        
        # LLM Configuration
        st.subheader("🤖 LLM Settings")
        llm_provider = st.selectbox(
            "LLM Provider",
            options=list(SQLAgentManager.SUPPORTED_LLM_PROVIDERS.keys()),
            help="Select your LLM provider"
        )
        
        with st.expander("LLM Configuration", expanded=True):
            if llm_provider == "OpenAI":
                model_options = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
            elif llm_provider == "Groq":
                model_options = ["mixtral-8x7b-32768", "llama2-70b-4096"]
            elif llm_provider == "Google":
                model_options = ["gemini-pro", "gemini-pro-vision"]
            else:
                model_options = ["custom-model"]
            
            model = st.selectbox("Model", options=model_options)
            api_key = st.text_input("API Key", type="password", help="Your API key for the selected provider")
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
            max_tokens = st.number_input("Max Tokens (optional)", min_value=1, value=None)
        
        if st.button("🚀 Initialize LLM", type="secondary"):
            if not api_key:
                st.error("Please provide API key")
            else:
                llm_config = LLMConfig(
                    provider=llm_provider,
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                with st.spinner("Initializing LLM..."):
                    if st.session_state.agent_manager.initialize_llm(llm_config):
                        st.session_state.llm_initialized = True
                        st.success("LLM initialized successfully!")
        
        st.markdown("---")
        
        if st.session_state.db_connected and st.session_state.llm_initialized:
            if st.button("⚡ Create SQL Agent", type="primary"):
                with st.spinner("Creating SQL Agent..."):
                    if st.session_state.agent_manager.create_agent():
                        st.session_state.agent_ready = True
                        st.success("SQL Agent ready!")
        
        # Status indicators
        st.markdown("### 📊 Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "✅" if st.session_state.db_connected else "❌"
            st.metric("Database", status)
        with col2:
            status = "✅" if st.session_state.llm_initialized else "❌"
            st.metric("LLM", status)
        with col3:
            status = "✅" if st.session_state.agent_ready else "❌"
            st.metric("Agent", status)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Query Interface")
        
        if st.session_state.agent_ready:
            question = st.text_area(
                "Enter your SQL question:",
                placeholder="e.g., Give me the list of all users who didn't login today along with their company name",
                height=100
            )
            
            col_query1, col_query2 = st.columns([1, 4])
            with col_query1:
                execute_query = st.button("🔍 Execute Query", type="primary")
            with col_query2:
                clear_history = st.button("🗑️ Clear History", type="secondary")
                if clear_history:
                    st.session_state.query_history = []
                    st.rerun()
            
            if execute_query and question:
                with st.spinner("Executing query..."):
                    result = st.session_state.agent_manager.execute_query(question)
                    st.session_state.query_history.append(result)
                    st.rerun()
            
            if st.session_state.query_history:
                st.subheader("📋 Query Results")
                latest_result = st.session_state.query_history[-1]
                if latest_result.get("error"):
                    st.error(f"Error: {latest_result['error']}")
                else:
                    st.success("Query executed successfully!")
                    if latest_result.get("final_answer"):
                        st.markdown("**Answer:**")
                        st.markdown(latest_result["final_answer"])
                    with st.expander("View Execution Steps", expanded=False):
                        for i, step in enumerate(latest_result.get("steps", []), 1):
                            st.markdown(f"**Step {i} ({step['type']}):**")
                            st.code(step["content"], language="text")
        else:
            st.info("Please configure database and LLM settings in the sidebar to get started.")
            st.subheader("🚀 Setup Progress")
            progress_steps = [
                ("Configure Database", st.session_state.db_connected),
                ("Initialize LLM", st.session_state.llm_initialized),
                ("Create Agent", st.session_state.agent_ready)
            ]
            for step_name, completed in progress_steps:
                status = "✅" if completed else "⏳"
                st.markdown(f"{status} {step_name}")
    
    with col2:
        st.header("📊 Database Info")
        if st.session_state.db_connected:
            if st.button("🔍 Refresh Table Info"):
                table_info = st.session_state.agent_manager.get_table_info()
                st.session_state.table_info = table_info
            
            if hasattr(st.session_state, 'table_info'):
                info = st.session_state.table_info
                if "error" not in info:
                    st.metric("Database Type", info.get("dialect", "Unknown"))
                    st.metric("Total Tables", info.get("total_tables", 0))
                    with st.expander("Available Tables", expanded=True):
                        for table in info.get("tables", [])[:20]:
                            st.text(f"• {table}")
                    with st.expander("Sample Schemas", expanded=False):
                        for table, schema in info.get("sample_schemas", {}).items():
                            st.markdown(f"**{table}:**")
                            st.code(schema, language="sql")
                else:
                    st.error(info["error"])
        
        if st.session_state.query_history:
            st.subheader("📝 Query History")
            for i, result in enumerate(reversed(st.session_state.query_history[-5:]), 1):
                with st.expander(f"Query {len(st.session_state.query_history) - i + 1}", expanded=False):
                    st.markdown(f"**Question:** {result['question']}")
                    st.markdown(f"**Time:** {result['timestamp']}")
                    if result.get('error'):
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success("Success")

if __name__ == "__main__":
    create_streamlit_ui()