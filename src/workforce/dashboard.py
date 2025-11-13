"""
Workforce Planning Dashboard UI

Provides Streamlit interface for workforce analytics and planning.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from .recommender import WorkforceRecommender
from .neo4j_connector import connect_to_neo4j


def show_workforce_planning():
    """Main function to display workforce planning interface."""
    st.title("🎯 AI Workforce Planning & Skills Analytics")
    st.markdown("**Intelligent skills gap analysis, career path recommendations, and internal talent mobility**")

    graph = connect_to_neo4j()
    if graph is None:
        st.warning("⚠️ Unable to connect to Neo4j database. Please check your configuration.")
        return

    recommender = WorkforceRecommender(graph)

    # Sub-navigation within workforce planning
    st.sidebar.markdown("---")
    st.sidebar.subheader("Workforce Analytics")
    page = st.sidebar.radio(
        "Analytics Modules:",
        ["📊 Data Overview", "🎯 Career Path Recommendations", "🔍 Skills Market Analysis", "🤝 Internal Talent Mobility"],
        key="workforce_page"
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
    """Display internal talent mobility interface."""
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
    """Display recruitment results."""
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
    """Returns advice based on compatibility level."""
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
    """Display career path recommendations interface."""
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
    """Display career recommendation results."""
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
    """Display workforce data overview."""
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
    """Display skills market analysis."""
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

