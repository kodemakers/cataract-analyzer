import os
import PyPDF2
import json
from typing import List, Dict, Any
import google.generativeai as genai
import concurrent.futures
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini API with direct key
API_KEY = "AIzaSyAMKfVohR-YRWP-f0jWCTN0KJ_VzgpA6kc"
genai.configure(api_key=API_KEY)

class ContractAnalyzer:
    def __init__(self, api_key: str = API_KEY, model_name: str = "gemini-2.0-flash", chunk_size: int = 10000, max_workers: int = 3):
        """Initialize the ContractAnalyzer with the specified API key and model."""
        # Configure with the provided API key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file using optimized extraction."""
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Use list comprehension and join for better performance
                text = "".join([page.extract_text() or "" for page in reader.pages])
                
                logger.info(f"Successfully extracted {len(text)} characters from PDF")
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of specified size to handle large contracts."""
        chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        logger.info(f"Split contract into {len(chunks)} chunks for processing")
        return chunks
    
    def analyze_chunk(self, chunk: str, chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """Analyze a single chunk of contract text."""
        try:
            logger.info(f"Analyzing chunk {chunk_index+1} of {total_chunks}")
            
            # Prepare analysis prompt - moved outside the loop for efficiency
            analysis_prompt = """
            You are a legal contract analysis expert. Analyze the following contract text and identify all risk elements.
            
            Categorize each identified risk into one of three categories:
            1. HIGH RISK: Critical issues that could lead to significant financial loss, legal liability, or business disruption
            2. MEDIUM RISK: Important issues that should be addressed but pose less immediate threat
            3. LOW RISK: Minor concerns that should be noted but are less likely to cause significant problems
            
            For each risk, provide:
            - Risk category (HIGH, MEDIUM, LOW)
            - Brief title/description
            - Contract section/clause reference
            - Explanation of the risk
            - Recommendation for mitigation
            
            Format your response as a structured JSON with the following schema:
            {
                "high_risks": [
                    {
                        "title": "Risk title",
                        "section": "Section reference",
                        "explanation": "Detailed explanation",
                        "mitigation": "Recommendation"
                    }
                ],
                "medium_risks": [...],
                "low_risks": [...]
            }
            
            CONTRACT TEXT:
            """
            
            # Try different model configurations if needed
            try:
                # First attempt with normal configuration
                logger.info(f"Attempting to analyze chunk {chunk_index+1} with primary configuration")
                response = self.model.generate_content(analysis_prompt + chunk)
                result_text = response.text
            except Exception as e:
                logger.error(f"Error with primary model configuration: {e}")
                
                # Try with different model parameter settings
                try:
                    logger.info(f"Trying alternative model configuration for chunk {chunk_index+1}")
                    # Create a new model instance with different settings
                    alt_model = genai.GenerativeModel(
                        self.model._model_name,
                        generation_config={"temperature": 0.1, "max_output_tokens": 1024}
                    )
                    response = alt_model.generate_content(analysis_prompt + chunk)
                    result_text = response.text
                except Exception as e2:
                    logger.error(f"Error with alternative model configuration: {e2}")
                    
                    # Last attempt with reduced content
                    logger.info(f"Trying with reduced content size for chunk {chunk_index+1}")
                    reduced_chunk = chunk[:len(chunk)//2]
                    response = self.model.generate_content(analysis_prompt + reduced_chunk)
                    result_text = response.text
            
            # Find JSON content (in case the model includes additional text)
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start == -1 or json_end <= json_start:
                logger.warning(f"JSON not found in response for chunk {chunk_index+1}")
                # Adding fallback response handling
                # Check if we can identify any risks in the text based on markers
                if "HIGH RISK" in result_text or "MEDIUM RISK" in result_text or "LOW RISK" in result_text:
                    logger.info(f"Attempting to parse non-JSON response with risk markers")
                    return self._extract_risks_from_text(result_text)
                return {"high_risks": [], "medium_risks": [], "low_risks": []}
                
            json_content = result_text[json_start:json_end]
            
            try:
                # Parse the JSON results
                chunk_results = json.loads(json_content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for chunk {chunk_index+1}: {e}")
                logger.error(f"JSON content: {json_content[:200]}...")
                
                # Try to fix common JSON formatting issues
                try:
                    # Replace single quotes with double quotes
                    fixed_json = json_content.replace("'", '"')
                    # Ensure proper quoting of keys
                    for key in ["title", "section", "explanation", "mitigation", "high_risks", "medium_risks", "low_risks"]:
                        fixed_json = fixed_json.replace(f"{key}:", f'"{key}":')
                    chunk_results = json.loads(fixed_json)
                    logger.info(f"Successfully fixed and parsed JSON for chunk {chunk_index+1}")
                except:
                    # If still can't parse, try text extraction
                    logger.info(f"Attempting text extraction fallback for chunk {chunk_index+1}")
                    return self._extract_risks_from_text(result_text)
            
            # Print the extracted JSON to console
            print(f"\n----- EXTRACTED JSON FROM CHUNK {chunk_index+1}/{total_chunks} -----")
            print(json.dumps(chunk_results, indent=2))
            print(f"----- END OF CHUNK {chunk_index+1} JSON -----\n")
            
            logger.info(f"Successfully analyzed chunk {chunk_index+1}")
            
            # Ensure the results have the expected structure
            if not isinstance(chunk_results, dict):
                chunk_results = {"high_risks": [], "medium_risks": [], "low_risks": []}
            for key in ["high_risks", "medium_risks", "low_risks"]:
                if key not in chunk_results:
                    chunk_results[key] = []
                elif not isinstance(chunk_results[key], list):
                    chunk_results[key] = []
            
            return chunk_results
        except Exception as e:
            logger.error(f"Error analyzing chunk {chunk_index+1}: {e}")
            # Return empty results in case of error to prevent overall analysis failure
            return {"high_risks": [], "medium_risks": [], "low_risks": []}
    
    def _extract_risks_from_text(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """Extract risks from non-JSON formatted text by looking for risk markers."""
        result = {"high_risks": [], "medium_risks": [], "low_risks": []}
        
        # Split by risk level markers
        high_risk_sections = text.split("HIGH RISK")[1:] if "HIGH RISK" in text else []
        medium_risk_sections = text.split("MEDIUM RISK")[1:] if "MEDIUM RISK" in text else []
        low_risk_sections = text.split("LOW RISK")[1:] if "LOW RISK" in text else []
        
        # Process each high risk section
        for section in high_risk_sections:
            end_idx = section.find("MEDIUM RISK") if "MEDIUM RISK" in section else (
                       section.find("LOW RISK") if "LOW RISK" in section else len(section))
            section = section[:end_idx].strip()
            
            # Extract basic info
            lines = [line.strip() for line in section.split("\n") if line.strip()]
            if not lines:
                continue
                
            title = lines[0] if lines else "Unspecified High Risk"
            section_ref = ""
            explanation = ""
            mitigation = ""
            
            # Look for section reference
            for line in lines:
                if "Section" in line or "Clause" in line or "Article" in line:
                    section_ref = line
                    break
            
            # Extract explanation and mitigation if possible
            explanation_idx = -1
            mitigation_idx = -1
            
            for i, line in enumerate(lines):
                if "explanation" in line.lower() or "details" in line.lower():
                    explanation_idx = i
                if "mitigation" in line.lower() or "recommendation" in line.lower():
                    mitigation_idx = i
            
            if explanation_idx >= 0 and mitigation_idx > explanation_idx:
                explanation = " ".join(lines[explanation_idx+1:mitigation_idx])
            elif explanation_idx >= 0:
                explanation = " ".join(lines[explanation_idx+1:])
            
            if mitigation_idx >= 0:
                mitigation = " ".join(lines[mitigation_idx+1:])
            
            result["high_risks"].append({
                "title": title,
                "section": section_ref,
                "explanation": explanation,
                "mitigation": mitigation
            })
        
        # Similar processing for medium and low risks...
        # (Processing code for medium and low risks would be similar to high risks)
        
        return result
    
    def analyze_contract(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze a contract PDF and categorize risks into high, medium, and low levels.
        Uses parallel processing for faster analysis.
        
        Args:
            pdf_path: Path to the PDF contract file
            
        Returns:
            Dictionary containing analysis results with risk categories
        """
        # Extract text from PDF
        contract_text = self.extract_text_from_pdf(pdf_path)
        
        # Split into manageable chunks if necessary
        chunks = self.chunk_text(contract_text)
        
        # Initialize results dictionary
        all_results = {
            "high_risks": [],
            "medium_risks": [],
            "low_risks": []
        }
        
        # Use ThreadPoolExecutor for parallel processing of chunks
        logger.info(f"Starting parallel analysis with {self.max_workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks and create a dictionary mapping futures to their indices
            future_to_index = {
                executor.submit(self.analyze_chunk, chunk, i, len(chunks)): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Process results as they complete with a progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(chunks), desc="Analyzing"):
                chunk_index = future_to_index[future]
                try:
                    chunk_results = future.result()
                    
                    # Merge results
                    all_results["high_risks"].extend(chunk_results.get("high_risks", []))
                    all_results["medium_risks"].extend(chunk_results.get("medium_risks", []))
                    all_results["low_risks"].extend(chunk_results.get("low_risks", []))
                    
                except Exception as e:
                    logger.error(f"Error processing results from chunk {chunk_index+1}: {e}")
        
        # Remove duplicate risks based on title and section
        all_results = self._deduplicate_risks(all_results)
        
        # Generate summary statistics
        summary = {
            "filename": os.path.basename(pdf_path),
            "total_risks": len(all_results["high_risks"]) + len(all_results["medium_risks"]) + len(all_results["low_risks"]),
            "high_risk_count": len(all_results["high_risks"]),
            "medium_risk_count": len(all_results["medium_risks"]),
            "low_risk_count": len(all_results["low_risks"]),
            "analysis": all_results
        }
        
        return summary
    
    def _deduplicate_risks(self, results: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
        """Remove duplicate risks based on title and section."""
        deduped_results = {"high_risks": [], "medium_risks": [], "low_risks": []}
        
        for risk_level in ["high_risks", "medium_risks", "low_risks"]:
            # Create a set to track seen risks
            seen_risks = set()
            
            for risk in results[risk_level]:
                # Create a tuple of title and section for uniqueness check
                risk_key = (risk.get("title", ""), risk.get("section", ""))
                
                # Only add if not seen before
                if risk_key not in seen_risks:
                    seen_risks.add(risk_key)
                    deduped_results[risk_level].append(risk)
        
        logger.info(f"Deduplicated risks - removed {sum(len(results[k]) for k in results) - sum(len(deduped_results[k]) for k in deduped_results)} duplicates")
        return deduped_results
    
    def save_analysis(self, analysis: Dict[str, Any], output_path: str) -> None:
        """Save analysis results to a JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Analysis saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving analysis to {output_path}: {e}")
            raise

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Analyze contract PDF using Gemini AI')
    parser.add_argument('pdf_path', help='Path to the contract PDF file')
    parser.add_argument('--api-key', default=API_KEY, help='Google Gemini API Key')
    parser.add_argument('--output', '-o', default=None, help='Output JSON file path')
    parser.add_argument('--chunk-size', type=int, default=4000, 
                        help='Size of text chunks for processing (default: 4000)')
    parser.add_argument('--workers', type=int, default=2, 
                        help='Number of parallel workers (default: 2)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--print-json', '-p', action='store_true', 
                        help='Print full JSON results to console after analysis')
    parser.add_argument('--model', '-m', default=None, 
                        help='Gemini model to use (will auto-detect if not specified)')
    parser.add_argument('--list-models', action='store_true',
                        help='List available Gemini models and exit')
    args = parser.parse_args()
    
    # Configure API
    genai.configure(api_key=args.api_key)
    
    # List available models if requested or if no model is specified
    if args.list_models or args.model is None:
        print("Listing available Gemini models...")
        try:
            models = genai.list_models()
            print("\nAVAILABLE MODELS:")
            model_names = []
            for model in models:
                print(f"- {model.name}")
                model_names.append(model.name)
                if hasattr(model, 'supported_generation_methods'):
                    print(f"  Supported methods: {model.supported_generation_methods}")
            
            # If just listing models, exit here
            if args.list_models:
                return
                
            # Auto-select model if none specified
            if args.model is None:
                # Try to find a text generation model
                text_models = [m for m in model_names if 'generateContent' in getattr(m, 'supported_generation_methods', [])]
                
                # If no specific text models found, try these common patterns
                potential_models = [
                    name for name in model_names if any(pattern in name.lower() 
                    for pattern in ['text', 'pro', 'gemini', 'palm'])
                ]
                
                if text_models:
                    args.model = text_models[0]
                elif potential_models:
                    args.model = potential_models[0]
                else:
                    # Fallback to a common model name pattern
                    for name in model_names:
                        if 'gemini' in name.lower():
                            args.model = name
                            break
                
                if args.model:
                    print(f"\nAutomatically selected model: {args.model}")
                else:
                    print("\nCould not automatically determine a suitable model.")
                    print("Please specify a model with --model from the list above.")
                    return
            
        except Exception as e:
            print(f"Error listing models: {e}")
            if args.list_models:
                return
            print("\nTrying common model names...")
            # Try some common model names
            common_models = ["models/gemini-pro", "gemini-pro", "text-bison", "models/text-bison"]
            args.model = common_models[0]  # Default to first one
    
    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set default output path if not specified
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        args.output = f"{base_name}_analysis.json"
    
    # Create analyzer with specified parameters
    analyzer = ContractAnalyzer(
        api_key=args.api_key,
        model_name=args.model,
        chunk_size=args.chunk_size,
        max_workers=args.workers
    )
    
    try:
        logger.info(f"Starting contract analysis for {args.pdf_path} using model {args.model}")
        start_time = __import__('time').time()
        
        analysis_results = analyzer.analyze_contract(args.pdf_path)
        analyzer.save_analysis(analysis_results, args.output)
        
        end_time = __import__('time').time()
        elapsed_time = end_time - start_time
        
        # Print summary
        print("\n===== CONTRACT ANALYSIS SUMMARY =====")
        print(f"File: {analysis_results['filename']}")
        print(f"Model used: {args.model}")
        print(f"Total risks identified: {analysis_results['total_risks']}")
        print(f"HIGH RISK items: {analysis_results['high_risk_count']}")
        print(f"MEDIUM RISK items: {analysis_results['medium_risk_count']}")
        print(f"LOW RISK items: {analysis_results['low_risk_count']}")
        print(f"Analysis completed in {elapsed_time:.2f} seconds")
        print("====================================\n")
        
        # Display top high risks if any
        if analysis_results['high_risk_count'] > 0:
            print("TOP HIGH RISK ITEMS:")
            for i, risk in enumerate(analysis_results['analysis']['high_risks'][:3]):
                print(f"{i+1}. {risk['title']} ({risk['section']})")
            print()
        
        # Print the full final JSON if requested
        if args.print_json:
            print("\n===== FULL ANALYSIS RESULTS =====")
            print(json.dumps(analysis_results, indent=2))
            print("=================================\n")
        
        print(f"Full analysis saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Error analyzing contract: {e}")
        import traceback
        traceback.print_exc()
        print("\nTIP: Try running with --list-models to see available models, then specify with --model")
        exit(1)

if __name__ == "__main__":
    main()