"""SQL agent factory used when the uploaded file is a database."""

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from .model_catalog import build_langchain_llm

def get_sql_agent(db_path: str, model: str, temperature: float):
    """Create a SQL agent bound to the uploaded SQLite database file."""
    # dynamic connection to the user's specific uploaded DB
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    
    llm = build_langchain_llm(model_id=model, temperature=temperature)
    
    return create_sql_agent(
        llm=llm,
        toolkit=SQLDatabaseToolkit(db=db, llm=llm),
        verbose=True,
        agent_type="    ",
        handle_parsing_errors=True
    )
