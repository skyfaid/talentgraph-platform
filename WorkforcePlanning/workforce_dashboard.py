# workforce_dashboard_xai_nomatplotlib.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from py2neo import Graph
import numpy as np
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings("ignore")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Workforce Planning & Skills Analytics",
    page_icon="🎯",
    layout="wide"
)

# --- NEO4J CONNECTION ---
def connect_to_neo4j():
    try:
        graph = Graph(
            "neo4j+s://f2e325e6.databases.neo4j.io",
            auth=("neo4j", "Tas8DQsuJnRLyWSXJpb6DH9weNBlE5rRZdG-d2NfZPs")
        )
        graph.run("RETURN 1 AS test")
        st.sidebar.success("✅ Connected to Neo4j Database")
        return graph
    except Exception as e:
        st.error(f"❌ Failed to connect to Neo4j: {e}")
        return None

# --- MAIN CLASS ---
class WorkforceRecommender:
    def __init__(self, graph):
        self.graph = graph

    def get_employee_profile(self, current_skills, experience_years=3, current_role=""):
        return {
            'current_skills': [s.lower().strip() for s in current_skills],
            'experience_years': experience_years,
            'current_role': current_role
        }

    def smart_recommendation(self, employee_profile, target_category):
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
            return {"status": "already_qualified", "message": "🎉 You already have all the required skills for this role!"}

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
        scores = []
        for skill in missing_skills:
            demand_score = self._get_skill_demand(skill)
            similarity_score = self._calculate_similarity(skill, current_skills)
            total_score = demand_score * 0.7 + similarity_score * 0.3
            scores.append((skill, total_score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s for s, sc in scores]

    def _get_skill_demand(self, skill):
        query = """
        MATCH (s:Skill {skill_name: $skill})<-[:REQUIRES]-(j:Job)
        RETURN COUNT(j) as demand
        """
        result = self.graph.run(query, skill=skill).data()
        demand = result[0]['demand'] if result else 0
        return min(demand / 1000, 1.0)

    def _calculate_similarity(self, target_skill, current_skills):
        if not current_skills:
            return 0
        
        # First check if RELATED_TO relationships exist in the database
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
            return max_similarity * 0.5  # Reduce fallback impact
        
        # If relationships exist, run normal query
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

    # --- LIME EXPLAINABILITY ---
    def explain_skill_priority(self, missing_skills, current_skills):
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
        """Calculate cosine similarity between two vectors"""
        if vec1 is None or vec2 is None:
            return 0.0
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    # --- CORRECTED TRANSFERABILITY CALCULATION ---
    def _calculate_transferability(self, employee_name, target_category, employee_skills):
        """Calculate transferability with multiple approaches"""
        
        # APPROACH 1: Check direct RELATED_TO relationships
        try:
            # Count how many employee skills are related to required skills in target category
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

        # APPROACH 3: Check via common skills in other jobs of target category
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
        """Basic text similarity"""
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

    # --- INTELLIGENT RECRUITMENT WITH TOP 5 EMPLOYEES ---
    def recruit_best_employees(self, target_category, min_threshold=0.15, top_n=5):
        """
        Find top N most suitable employees for a target category
        """
        # 1) Get TOP 50 target skills
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

        # 2) Get ALL employees
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

        # List to store all employees with their scores
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
                # Use embeddings if available
                similarity_matrix = np.zeros((len(target_embeddings), len(emp_embeddings)))
                for i, target_emb in enumerate(target_embeddings):
                    for j, emp_emb in enumerate(emp_embeddings):
                        similarity_matrix[i, j] = self._cosine_similarity(target_emb, emp_emb)
                
                best_similarities = np.max(similarity_matrix, axis=1)
                semantic_similarity = np.mean(best_similarities)
            else:
                # Fallback: text similarity
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

    
# --- MAIN INTERFACE ---
def main():
    st.title("🎯 AI Workforce Planning & Skills Analytics Platform")
    st.markdown("**Intelligent skills gap analysis, career path recommendations, and internal talent mobility**")

    graph = connect_to_neo4j()
    if graph is None:
        st.stop()

    recommender = WorkforceRecommender(graph)

    st.sidebar.title("Platform Navigation")
    page = st.sidebar.radio(
        "Analytics Modules:",
        ["📊 Data Overview", "🎯 Career Path Recommendations", "🔍 Skills Market Analysis", "🤝 Internal Talent Mobility"]
    )

    try:
        categories_data = graph.run("MATCH (j:Job) RETURN DISTINCT j.category as cat ORDER BY cat").data()
        category_list = [c['cat'] for c in categories_data]
    except:
        category_list = ["INFORMATION-TECHNOLOGY", "BUSINESS-DEVELOPMENT", "FINANCE", "SALES", "HR"]

    if page == "📊 Data Overview":
        show_overview(graph)
    elif page == "🎯 Career Path Recommendations":
        show_recommendations(graph, recommender, category_list)
    elif page == "🔍 Skills Market Analysis":
        show_market_analysis(graph, category_list)
    elif page == "🤝 Internal Talent Mobility":
        show_recruitment(recommender, category_list)

def show_recruitment(recommender, category_list):
    st.header("🤝 Internal Talent Mobility & Redeployment")
    st.markdown("**Identify existing employees with transferable skills for new roles**")
    
    col1, col2 = st.columns(2)
    with col1:
        target_category = st.selectbox("Target Department/Role Category", category_list)
        
    with col2:
        min_thresh = st.slider("Minimum Compatibility Score (%)", 0, 100, 15)
        min_thresh_float = min_thresh / 100.0
        
        top_n = st.slider("Number of Top Candidates to Display", 1, 10, 5)

        st.markdown("---")
        st.markdown("**💡 How it works:** Analyzes skill compatibility, semantic similarity, and transferability potential")

    if st.button("🔍 Find Best Internal Candidates"):
        with st.spinner(f"Analyzing employee skills and compatibility for {target_category} roles..."):
            profiles = recommender.recruit_best_employees(target_category, min_threshold=min_thresh_float, top_n=top_n)

        display_recruitment_results(profiles, min_thresh, top_n)

def display_recruitment_results(profiles, min_thresh, top_n):
    if profiles:
        st.success(f"🎯 **Found {len(profiles)} qualified internal candidates** (showing top {len(profiles)} of {top_n} requested)")
        
        # Display each profile in an expander
        for i, profile in enumerate(profiles, 1):
            with st.expander(f"🏆 Rank #{i}: {profile['name']} - {profile['compatibility']}% Overall Match", expanded=(i==1)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Current Position:** {profile['role']}")
                    st.write(f"**Current Department:** {profile['current_category']}")
                    
                with col2:
                    st.metric("Overall Compatibility", f"{profile['compatibility']}%")
                
                # Detailed metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Skills Coverage", f"{profile['coverage_target']}%", 
                           delta=f"{profile['exact_matches']}/{profile['total_target_skills']} skills matched")
                col2.metric("Semantic Similarity", f"{profile['semantic_similarity']}%")
                col3.metric("Skill Transferability", f"{profile['transferability']}%")
                col4.metric("Exact Skill Matches", profile['exact_matches'])
                
                # Skills
                st.write("🔧 **Employee's Current Skills:**")
                skills_text = ", ".join(profile['skills'][:15])
                st.write(skills_text)
                if len(profile['skills']) > 15:
                    st.write(f"... and {len(profile['skills']) - 15} additional skills")
                
                # Compatibility advice
                st.write(get_compatibility_advice(profile['compatibility']))
                
                st.markdown("---")

        # Global summary
        st.subheader("📊 Candidate Comparison Summary")
        summary_data = []
        for profile in profiles:
            summary_data.append({
                'Candidate': profile['name'],
                'Current Role': profile['role'],
                'Current Department': profile['current_category'],
                'Overall Match %': profile['compatibility'],
                'Exact Skills': profile['exact_matches'],
                'Skills Coverage %': profile['coverage_target'],
                'Semantic Similarity %': profile['semantic_similarity'],
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)

    else:
        st.error(f"**No internal candidates meet the {min_thresh}% compatibility threshold**")
        st.info("""
        **Recommended Actions:**
        - 📉 Consider lowering the compatibility threshold to 10-15%
        - 🔍 Explore external recruitment options
        - 🎯 Develop targeted upskilling programs for current employees
        - 📚 Identify skill gaps and create training initiatives
        """)

def get_compatibility_advice(compatibility):
    """Returns advice based on compatibility level"""
    if compatibility < 15:
        return "🔴 **Low Compatibility** - Consider for long-term development program (12+ months) or external hiring"
    elif compatibility < 30:
        return "🟡 **Moderate Potential** - Requires significant training (6-12 months) but shows transferable skills"
    elif compatibility < 50:
        return "🔵 **Good Potential** - Suitable with moderate training investment (3-6 months)"
    elif compatibility < 70:
        return "🟢 **Strong Match** - Ready for transition with minimal training (1-3 months)"
    else:
        return "🎉 **Excellent Match** - Ideal candidate for immediate transition or promotion"

def show_recommendations(graph, recommender, category_list):
    st.header("🎯 Personalized Career Path Recommendations")
    st.markdown("**Identify skill gaps and recommended learning paths for career advancement**")
    
    col1, col2 = st.columns(2)
    with col1:
        current_role = st.text_input("Your Current Role/Position", "Data Analyst")
        experience_years = st.slider("Years of Professional Experience", 0, 30, 3)
    with col2:
        current_skills_input = st.text_area(
            "Your Current Skills (comma separated)",
            "python, sql, data analysis, communication, problem solving, excel, statistics",
            help="List all your relevant skills separated by commas"
        )
        current_skills = [s.strip() for s in current_skills_input.split(',') if s.strip()]
    
    st.markdown("---")
    target_category = st.selectbox("Target Career Path / Department", category_list, key="reco_category")
    
    if st.button("🚀 Analyze My Career Path"):
        if not current_skills:
            st.warning("Please enter your current skills to get personalized recommendations")
            return
            
        with st.spinner("Analyzing your profile against market requirements..."):
            profile = recommender.get_employee_profile(current_skills, experience_years, current_role)
            recommendations = recommender.smart_recommendation(profile, target_category)
            display_recommendation_results(recommendations, profile, recommender)

def display_recommendation_results(recommendations, profile, recommender):
    st.subheader("📊 Career Transition Analysis")
    
    if recommendations['status'] == 'already_qualified':
        st.success(recommendations['message'])
        st.info("**Next Steps:** Consider applying for senior roles or specialized positions in this field")
        return
        
    col1, col2, col3 = st.columns(3)
    col1.metric("Skills Gap", recommendations['missing_skills_count'])
    col2.metric("Estimated Training Timeline", f"{recommendations['duration_months']} months")
    col3.metric("Recommendation Confidence", recommendations['confidence'].title())

    st.subheader("📈 Priority Skills to Develop")
    st.markdown("**Focus on these skills in order of importance:**")
    
    df = pd.DataFrame({
        'Skill': [skill.title() for skill in recommendations['priority_skills']],
        'Priority Level': range(1, len(recommendations['priority_skills'])+1),
        'Learning Order': ['Immediate Focus' if i == 1 else f'Phase {i}' for i in range(1, len(recommendations['priority_skills'])+1)]
    })
    st.dataframe(df, use_container_width=True)

    st.subheader("🧠 Explanation: Why These Skills?")
    st.markdown("**Understanding the reasoning behind skill priorities:**")
    
    xai_expl = recommender.explain_skill_priority(recommendations['priority_skills'], profile['current_skills'])
    for e in xai_expl:
        with st.expander(f"🔍 {e['skill'].title()} - Priority Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Market Demand Score", f"{e['demand_score']:.2f}")
            with col2:
                st.metric("Similarity to Your Skills", f"{e['similarity_score']:.2f}")
            
            st.write("**Factor Contributions:**")
            for factor, weight in e['lime_explanation'].items():
                if "Market Demand" in factor:
                    st.write(f"📊 {factor}: {weight:.3f}")
                else:
                    st.write(f"🔄 {factor}: {weight:.3f}")

def show_overview(graph):
    st.header("📊 Workforce Data Overview")
    st.markdown("**Comprehensive view of skills, roles, and organizational capabilities**")
    
    try:
        jobs_count = graph.run("MATCH (j:Job) RETURN count(j) as count").data()[0]['count']
        skills_count = graph.run("MATCH (s:Skill) RETURN count(s) as count").data()[0]['count']
        rels_count = graph.run("MATCH ()-[r:REQUIRES]->() RETURN count(r) as count").data()[0]['count']
        related_count = graph.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count").data()[0]['count']
        employees_count = graph.run("MATCH (e:Employee) RETURN count(e) as count").data()[0]['count']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Job Roles", jobs_count)
        col2.metric("Unique Skills", skills_count)
        col3.metric("Employees", employees_count)
        col4.metric("Skill Requirements", rels_count)
        col5.metric("Skill Relationships", related_count)

        # Top skills chart
        st.subheader("🏆 Most In-Demand Skills Across Organization")
        top_skills = graph.run("""
            MATCH (s:Skill)<-[:REQUIRES]-(j:Job)
            RETURN s.skill_name as skill, COUNT(j) as demand
            ORDER BY demand DESC
            LIMIT 10
        """).data()
        
        if top_skills:
            df = pd.DataFrame(top_skills)
            fig = px.bar(df, x='demand', y='skill', orientation='h', 
                        title="Top 10 Most Required Skills",
                        labels={'demand': 'Number of Roles Requiring Skill', 'skill': 'Skill Name'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Department distribution
        st.subheader("📈 Role Distribution by Department")
        dept_dist = graph.run("""
            MATCH (j:Job)
            RETURN j.category as department, COUNT(j) as count
            ORDER BY count DESC
        """).data()
        
        if dept_dist:
            df_dept = pd.DataFrame(dept_dist)
            fig_dept = px.pie(df_dept, values='count', names='department', 
                             title="Job Distribution Across Departments")
            st.plotly_chart(fig_dept, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading overview data: {e}")

def show_market_analysis(graph, category_list):
    st.header("🔍 Skills Market Analysis")
    st.markdown("**Explore skill requirements and trends across different departments**")
    
    selected_category = st.selectbox("Select Department to Analyze", category_list, key="market_category")
    
    if selected_category:
        with st.spinner(f"Analyzing skill requirements for {selected_category}..."):
            skills = graph.run("""
                MATCH (j:Job {category:$cat})-[:REQUIRES]->(s:Skill)
                RETURN s.skill_name as skill, COUNT(j) as demand
                ORDER BY demand DESC
                LIMIT 15
            """, cat=selected_category).data()
            
        if skills:
            df = pd.DataFrame(skills)
            
            st.subheader(f"📊 Top Skills Required in {selected_category}")
            
            fig = px.bar(df, x='demand', y='skill', orientation='h',
                        title=f"Most Required Skills in {selected_category}",
                        labels={'demand': 'Number of Roles', 'skill': 'Skill Name'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 Detailed Skills Breakdown")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning(f"No skill data found for {selected_category}")

if __name__ == "__main__":
    main()