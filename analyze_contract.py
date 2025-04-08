import tempfile
import streamlit as st
from analyzer import ContractAnalyzer
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Contract Risk Analyzer",
    page_icon="📄",
    layout="wide"
)
st.image('kode_logo.png', width=200)

# App title and description
st.title("Contract Risk Analyzer")
st.markdown("""
This application uses Gemini AI to analyze contract documents and identify potential risks.
Upload your contract PDF to get a detailed risk analysis categorized by severity.
""")

# Sidebar for API key input
# with st.sidebar:
#     st.header("Configuration")
#     api_key = st.text_input("Enter your Google API Key", type="password")
    
#     st.markdown("---")
#     st.markdown("### About")
#     st.markdown("""
#     This tool analyzes contracts and categorizes risks into:
#     - 🔴 **High Risk** - Critical issues requiring immediate attention
#     - 🟠 **Medium Risk** - Important issues to be addressed
#     - 🟢 **Low Risk** - Minor concerns to be noted
#     """)

# Main area for file upload and results
uploaded_file = st.file_uploader("Upload Contract PDF", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name
    
    # Check if API key is provided
    # if not api_key:
    #     st.error("Please enter your Google API Key in the sidebar to proceed.")
    # else:
    #     # Initialize the analyzer with direct API key
        analyzer = ContractAnalyzer(api_key='AIzaSyAMKfVohR-YRWP-f0jWCTN0KJ_VzgpA6kc')
        
        # Show progress during analysis
        with st.spinner("Analyzing contract... This may take a few minutes depending on the document size."):
            try:
                # Perform analysis
                analysis_results = analyzer.analyze_contract(temp_file_path)
                
                # Display summary metrics
                st.success("Analysis completed successfully!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("High Risk Items", analysis_results["high_risk_count"])
                col2.metric("Medium Risk Items", analysis_results["medium_risk_count"])
                col3.metric("Low Risk Items", analysis_results["low_risk_count"])
                
                # Create tabs for different risk levels
                high_tab, medium_tab, low_tab = st.tabs(["High Risk", "Medium Risk", "Low Risk"])
                
                # Display high risks
                with high_tab:
                    if analysis_results["high_risk_count"] == 0:
                        st.info("No high risk items identified.")
                    else:
                        for i, risk in enumerate(analysis_results["analysis"]["high_risks"]):
                            with st.expander(f"{i+1}. {risk['title']} ({risk['section']})"):
                                st.markdown(f"**Section:** {risk['section']}")
                                st.markdown(f"**Explanation:** {risk['explanation']}")
                                st.markdown(f"**Recommended Mitigation:** {risk['mitigation']}")
                
                # Display medium risks
                with medium_tab:
                    if analysis_results["medium_risk_count"] == 0:
                        st.info("No medium risk items identified.")
                    else:
                        for i, risk in enumerate(analysis_results["analysis"]["medium_risks"]):
                            with st.expander(f"{i+1}. {risk['title']} ({risk['section']})"):
                                st.markdown(f"**Section:** {risk['section']}")
                                st.markdown(f"**Explanation:** {risk['explanation']}")
                                st.markdown(f"**Recommended Mitigation:** {risk['mitigation']}")
                
                # Display low risks
                with low_tab:
                    if analysis_results["low_risk_count"] == 0:
                        st.info("No low risk items identified.")
                    else:
                        for i, risk in enumerate(analysis_results["analysis"]["low_risks"]):
                            with st.expander(f"{i+1}. {risk['title']} ({risk['section']})"):
                                st.markdown(f"**Section:** {risk['section']}")
                                st.markdown(f"**Explanation:** {risk['explanation']}")
                                st.markdown(f"**Recommended Mitigation:** {risk['mitigation']}")
                
                # Allow downloading the analysis as JSON
                st.download_button(
                    label="Download Full Analysis Report (JSON)",
                    data=str(analysis_results),
                    file_name=f"{uploaded_file.name}_analysis.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Error analyzing contract: {str(e)}")
            
            finally:
                # Clean up the temporary file
                import os
                os.unlink(temp_file_path)

else:
    # Display sample analysis or instructions when no file is uploaded
    st.info("Please upload a contract PDF file to begin analysis.")
    
    # Show example of what the output will look like
    with st.expander("View Sample Analysis"):
        st.markdown("""
        ### Sample Risk Analysis
        
        #### 🔴 High Risk Example
        **Title:** Unlimited Liability Clause  
        **Section:** Section 14.2  
        **Explanation:** The contract imposes unlimited liability for any breach, with no caps or exclusions.  
        **Mitigation:** Negotiate liability caps and specific exclusions for indirect damages.
        
        #### 🟠 Medium Risk Example
        **Title:** Vague Termination Rights  
        **Section:** Section 9.3  
        **Explanation:** Termination rights are ambiguously defined, creating uncertainty about when the contract can be terminated.  
        **Mitigation:** Clarify specific conditions that would permit termination by either party.
        
        #### 🟢 Low Risk Example
        **Title:** Missing Contact Information  
        **Section:** Section 22.1  
        **Explanation:** Notice provision lacks specific email addresses for formal communications.  
        **Mitigation:** Update with current contact details for all official notices.
        """)

# Footer
st.markdown("---")
st.caption("Powered by Google Gemini AI")