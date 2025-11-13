"""
Workforce Planning Module

Provides workforce analytics, career path recommendations, and internal talent mobility.
"""
from .recommender import WorkforceRecommender
from .neo4j_connector import connect_to_neo4j
from .dashboard import show_workforce_planning

__all__ = ['WorkforceRecommender', 'connect_to_neo4j', 'show_workforce_planning']

