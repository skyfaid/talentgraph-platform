"""
Neo4j Database Connection for Workforce Planning
"""
import streamlit as st
from py2neo import Graph
from src.utils.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


def connect_to_neo4j():
    """Connect to Neo4j database and return graph instance."""
    try:
        graph = Graph(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        # Test connection
        graph.run("RETURN 1 AS test")
        return graph
    except Exception as e:
        st.error(f"❌ Failed to connect to Neo4j: {e}")
        return None

