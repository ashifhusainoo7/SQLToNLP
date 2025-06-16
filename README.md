# 🔍 Streamlit SQL Database Agent

A Streamlit application that leverages LangChain and various LLM providers (OpenAI, Groq, Google, Anthropic) to create a safe, production-grade SQL querying assistant. Easily connect to your SQL Server database, configure an LLM, and get natural language insights with query execution and analysis.

---

## 📋 Table of Contents

- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
  - [Database Settings](#database-settings)
  - [LLM Settings](#llm-settings)
- [Running the App](#-running-the-app)
- [Usage Guide](#-usage-guide)
  - [1. Configure Database](#1-configure-database)
  - [2. Initialize LLM](#2-initialize-llm)
  - [3. Create SQL Agent](#3-create-sql-agent)
  - [4. Execute Queries](#4-execute-queries)
  - [5. View Database Info & History](#5-view-database-info--history)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Features

- **One-click** connection testing to SQL Server via ODBC
- Support for multiple LLM providers: OpenAI, Groq, Google, Anthropic
- Safe SQL agent with enforced read-only rules
- Interactive, step-by-step query execution breakdown
- Live metadata and schema browsing
- Query history tracking and replay

---

## 📜 Prerequisites

- Python 3.8 or higher
- A SQL Server instance accessible over network
- API key for your chosen LLM provider (OpenAI, Groq, Google, or Anthropic)

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ashifhusainoo7/SQLToNLP.git
   cd SQLToNLP
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Configuration

All configuration is done through the Streamlit sidebar.

### Database Settings

1. **Server**: Your SQL Server address (e.g., `myserver\SQLEXPRESS`)
2. **Database**: Target database name
3. **Username** & **Password**: Credentials with read-only permissions
4. **Driver**: Choose ODBC Driver 17 or 18 for SQL Server (or FreeTDS)

Click **Test Database Connection** to verify and initialize.

### LLM Settings

1. **LLM Provider**: Select from OpenAI, Groq, Google, Anthropic
2. **Model**: Choose model compatible with provider (e.g., `gpt-4-turbo`)
3. **API Key**: Enter your provider API key
4. **Temperature** & **Max Tokens** (optional)

Click **Initialize LLM** to connect.

---

## ▶️ Running the App

Start the Streamlit server:

```bash
streamlit run streamlit_with_sql.py
```

This will open the UI in your default browser at `http://localhost:8501`.

---

## 🧭 Usage Guide

Once the app is running, follow these steps:

### 1. Configure Database

- Enter SQL Server details in the sidebar
- Click **🔗 Test Database Connection**
- Upon success, the Database status indicator turns ✅

### 2. Initialize LLM

- Select LLM provider and model
- Enter your API key and adjust parameters
- Click **🚀 Initialize LLM**
- LLM status indicator turns ✅

### 3. Create SQL Agent

- After database and LLM are ready, click **⚡ Create SQL Agent**
- Agent status indicator turns ✅

### 4. Execute Queries

- In the **Query Interface**, type natural language questions (e.g., "List users who haven't logged in today with their company names")
- Click **🔍 Execute Query** to send to the agent
- View results and the generated SQL in the "View Execution Steps" expander

### 5. View Database Info & History

- **Refresh Table Info** to load current schema: total tables, sample schemas
- Expand **Available Tables** to browse object names
- Query history appears in the sidebar with timestamps and statuses

---

## 🗂️ Project Structure

```
SQLToNLP/
├── streamlit_with_sql.py      # Main application file
├── requirements.txt          # Python dependencies
└── README.md                 # This documentation
```

---

## 📦 Dependencies

See [requirements.txt](requirements.txt) for the full list:

```text
streamlit
pandas
sqlalchemy
langchain
langchain-openai
langchain-anthropic
langchain-google-genai
langchain-groq
langgraph
langchain_community
pyodbc
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

