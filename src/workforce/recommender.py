"""
Workforce Recommender System

Provides intelligent skill gap analysis, career path recommendations,
and internal talent mobility analysis.
"""
import numpy as np
import lime
import lime.lime_tabular
import streamlit as st
from typing import List, Dict, Any


class WorkforceRecommender:
    """Intelligent workforce planning and recommendation system."""
    
    def __init__(self, graph):
        """Initialize with Neo4j graph connection."""
        self.graph = graph

    def get_employee_profile(self, current_skills, experience_years=3, current_role=""):
        """Create employee profile from input data."""
        return {
            'current_skills': [s.lower().strip() for s in current_skills],
            'experience_years': experience_years,
            'current_role': current_role
        }

    def smart_recommendation(self, employee_profile, target_category):
        """Generate smart career path recommendations."""
        current_skills = employee_profile['current_skills']
        query = """
        MATCH (j:Job {category: $category})-[:REQUIRES]->(s:Skill)
        RETURN s.skill_name as skill, COUNT(j) as demand
        ORDER BY demand DESC
        LIMIT 20
        """
        results = self.graph.run(query, category=target_category).data()
        target_skills = [r['skill'] for r in results]

        missing_skills = [s for s in target_skills if s not in current_skills]
        if not missing_skills:
            return {
                "status": "already_qualified",
                "message": "🎉 You already have all the required skills for this role!"
            }

        prioritized_skills = self._prioritize_skills(missing_skills, current_skills)
        duration = max(3, min(18, len(missing_skills)))

        return {
            "status": "reskilling_needed",
            "target": target_category,
            "missing_skills_count": len(missing_skills),
            "priority_skills": prioritized_skills[:5],
            "duration_months": duration,
            "confidence": "high"
        }

    def _prioritize_skills(self, missing_skills, current_skills):
        """Prioritize missing skills based on demand and similarity."""
        scores = []
        for skill in missing_skills:
            demand_score = self._get_skill_demand(skill)
            similarity_score = self._calculate_similarity(skill, current_skills)
            total_score = demand_score * 0.7 + similarity_score * 0.3
            scores.append((skill, total_score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s for s, sc in scores]

    def _get_skill_demand(self, skill):
        """Get market demand score for a skill."""
        query = """
        MATCH (s:Skill {skill_name: $skill})<-[:REQUIRES]-(j:Job)
        RETURN COUNT(j) as demand
        """
        result = self.graph.run(query, skill=skill).data()
        demand = result[0]['demand'] if result else 0
        return min(demand / 1000, 1.0)

    def _calculate_similarity(self, target_skill, current_skills):
        """Calculate similarity between target skill and current skills."""
        if not current_skills:
            return 0
        
        # Check if RELATED_TO relationships exist
        relation_check = self.graph.run("""
            MATCH ()-[:RELATED_TO]->() 
            RETURN count(*) as relation_count
            LIMIT 1
        """).data()
        
        relation_count = relation_check[0]['relation_count'] if relation_check else 0
        
        if relation_count == 0:
            # Fallback: text similarity
            max_similarity = 0.0
            for current_skill in current_skills:
                text_sim = self._text_similarity(target_skill, current_skill)
                if text_sim > max_similarity:
                    max_similarity = text_sim
            return max_similarity * 0.5
        
        # Use RELATED_TO relationships
        query = """
        MATCH (target:Skill {skill_name: $target_skill})
        MATCH (current:Skill)
        WHERE current.skill_name IN $current_skills
        MATCH (target)-[:RELATED_TO]-(current)
        RETURN COUNT(current) as similarity
        """
        result = self.graph.run(query, target_skill=target_skill, current_skills=current_skills).data()
        similarity = result[0]['similarity'] if result else 0
        return min(similarity / 5, 1.0)

    def explain_skill_priority(self, missing_skills, current_skills):
        """Generate LIME explanations for skill priorities."""
        explanations = []
        data = []
        features = []
        for skill in missing_skills:
            demand = self._get_skill_demand(skill)
            similarity = self._calculate_similarity(skill, current_skills)
            score = demand * 0.7 + similarity * 0.3
            data.append([demand, similarity, score])
            features.append(skill)
        if not data:
            return []

        X = np.array(data)[:, :2]
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X, feature_names=["Market Demand", "Skill Similarity"], mode="regression"
        )
        for i, skill in enumerate(features):
            lime_exp = lime_explainer.explain_instance(X[i], lambda x: x[:,0]*0.7 + x[:,1]*0.3)
            explanations.append({
                "skill": skill,
                "demand_score": X[i][0],
                "similarity_score": X[i][1],
                "lime_explanation": dict(lime_exp.as_list())
            })
        return explanations

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _calculate_transferability(self, employee_name, target_category, employee_skills):
        """Calculate transferability with multiple approaches."""
        
        # APPROACH 1: Check direct RELATED_TO relationships
        try:
            query1 = """
            MATCH (e:Employee {name: $ename})-[:HAS_SKILL]->(emp_skill:Skill)
            MATCH (emp_skill)-[:RELATED_TO]->(related_skill:Skill)
            MATCH (related_skill)<-[:REQUIRES]-(:Job {category: $cat})
            RETURN COUNT(DISTINCT emp_skill) as transfer_count
            """
            result1 = self.graph.run(query1, ename=employee_name, cat=target_category).data()
            
            if result1 and result1[0]['transfer_count'] > 0:
                transfer_count = result1[0]['transfer_count']
                return min(transfer_count / len(employee_skills), 1.0)
        except Exception as e:
            st.warning(f"Skill relationship analysis failed: {e}")

        # APPROACH 2: Check RELATED_TO relationships in both directions
        try:
            query2 = """
            MATCH (e:Employee {name: $ename})-[:HAS_SKILL]->(emp_skill:Skill)
            MATCH (emp_skill)-[:RELATED_TO]-(related_skill:Skill)
            MATCH (related_skill)<-[:REQUIRES]-(:Job {category: $cat})
            RETURN COUNT(DISTINCT emp_skill) as transfer_count
            """
            result2 = self.graph.run(query2, ename=employee_name, cat=target_category).data()
            
            if result2 and result2[0]['transfer_count'] > 0:
                transfer_count = result2[0]['transfer_count']
                return min(transfer_count / len(employee_skills), 1.0)
        except Exception as e:
            st.warning(f"Bidirectional skill analysis failed: {e}")

        # APPROACH 3: Check via common skills
        try:
            query3 = """
            MATCH (e:Employee {name: $ename})-[:HAS_SKILL]->(emp_skill:Skill)
            MATCH (other_job:Job {category: $cat})-[:REQUIRES]->(other_skill:Skill)
            WHERE emp_skill.skill_name = other_skill.skill_name
            RETURN COUNT(DISTINCT emp_skill) as common_skills_count
            """
            result3 = self.graph.run(query3, ename=employee_name, cat=target_category).data()
            
            if result3:
                common_count = result3[0]['common_skills_count']
                if common_count > 0:
                    return min(common_count * 0.1, 0.5)
        except Exception as e:
            st.warning(f"Common skills analysis failed: {e}")

        return 0.0

    def _text_similarity(self, skill1, skill2):
        """Basic text similarity calculation."""
        words1 = set(skill1.lower().split())
        words2 = set(skill2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Bonus for substrings
        substring_bonus = 0.0
        for word1 in words1:
            for word2 in words2:
                if word1 in word2 or word2 in word1:
                    substring_bonus += 0.2
                    
        return min(jaccard + substring_bonus, 1.0)

    def recruit_best_employees(self, target_category, min_threshold=0.15, top_n=5):
        """Find top N most suitable employees for a target category."""
        # Get TOP 50 target skills
        target_data = self.graph.run("""
            MATCH (j:Job {category:$cat})-[:REQUIRES]->(s:Skill)
            WITH s, COUNT(j) as demand
            ORDER BY demand DESC
            LIMIT 50
            RETURN s.skill_name as skill_name, s.embedding as embedding
        """, cat=target_category).data()

        if not target_data:
            return []

        target_skills = [s['skill_name'].lower() for s in target_data]
        target_embeddings = [np.array(s['embedding']) for s in target_data if s['embedding'] is not None]
        
        st.info(f"🔍 Analyzing {len(target_skills)} key skills for {target_category} roles")

        # Get ALL employees
        employees_data = self.graph.run("""
            MATCH (e:Employee)-[:HAS_SKILL]->(s:Skill)
            WHERE coalesce(e.category,'') <> $cat
            WITH e, collect({name: s.skill_name, embedding: s.embedding}) as skills
            RETURN e.name as name, 
                   e.role as role, 
                   e.category as category,
                   skills
            ORDER BY e.name
        """, cat=target_category).data()

        if not employees_data:
            return []

        st.info(f"👥 Evaluating {len(employees_data)} employees for internal mobility")

        all_employees_scores = []
        processed_count = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        for emp in employees_data:
            processed_count += 1
            progress_bar.progress(processed_count / len(employees_data))
            status_text.text(f"Analyzing skills match for {emp['name']} ({processed_count}/{len(employees_data)})")

            emp_skills_data = emp['skills']
            emp_skills = [s['name'].lower() for s in emp_skills_data]
            emp_embeddings = [np.array(s['embedding']) for s in emp_skills_data if s['embedding'] is not None]

            if not emp_skills:
                continue

            set_emp = set(emp_skills)
            set_target = set(target_skills)
            common = len(set_emp & set_target)

            # Coverage target
            coverage_target = common / max(1, len(set_target))

            # Semantic similarity
            semantic_similarity = 0.0
            
            if target_embeddings and emp_embeddings:
                similarity_matrix = np.zeros((len(target_embeddings), len(emp_embeddings)))
                for i, target_emb in enumerate(target_embeddings):
                    for j, emp_emb in enumerate(emp_embeddings):
                        similarity_matrix[i, j] = self._cosine_similarity(target_emb, emp_emb)
                
                best_similarities = np.max(similarity_matrix, axis=1)
                semantic_similarity = np.mean(best_similarities)
            else:
                similarities = []
                for target_skill in target_skills:
                    best_sim = 0.0
                    for emp_skill in emp_skills:
                        sim = self._text_similarity(target_skill, emp_skill)
                        if sim > best_sim:
                            best_sim = sim
                    similarities.append(best_sim)
                
                if similarities:
                    semantic_similarity = np.mean(similarities)
                    semantic_similarity = min(semantic_similarity * 1.5, 1.0)

            # TRANSFERABILITY
            transferability = self._calculate_transferability(emp['name'], target_category, emp_skills)

            # Combined score
            adjusted_coverage = min(coverage_target * 3, 1.0)
            
            score = (0.30 * adjusted_coverage) + (0.30 * semantic_similarity) + (0.20 * transferability)
            
            # Bonus for exact skills
            exact_bonus = min(common * 0.08, 0.4)
            score = min(score * (1.0 + exact_bonus), 1.0)

            # Add employee to list if above threshold
            if score >= min_threshold:
                employee_profile = {
                    "name": emp['name'],
                    "role": emp.get('role', ''),
                    "current_category": emp.get('category', ''),
                    "skills": emp_skills,
                    "compatibility": round(score * 100, 1),
                    "coverage_target": round(coverage_target * 100, 1),
                    "semantic_similarity": round(semantic_similarity * 100, 1),
                    "transferability": round(transferability * 100, 1),
                    "exact_matches": common,
                    "total_target_skills": len(target_skills),
                    "adjusted_coverage": round(adjusted_coverage * 100, 1),
                    "raw_score": score
                }
                all_employees_scores.append(employee_profile)

        progress_bar.empty()
        status_text.empty()

        # Sort by score and return TOP N
        all_employees_scores.sort(key=lambda x: x['raw_score'], reverse=True)
        return all_employees_scores[:top_n]

