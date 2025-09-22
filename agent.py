#!/usr/bin/env python3
"""
AI Agent for Bank Statement PDF Parsing

Architecture:
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Planner   │───▶│ Code Gen     │───▶│   Tester    │───▶│  Evaluator   │
│             │    │              │    │             │    │              │
│ - Analyze   │    │ - Generate   │    │ - Run tests │    │ - Check pass │
│   PDF       │    │   parser.py  │    │ - Compare   │    │ - Self-fix   │
│ - Extract   │    │ - Use LLM    │    │   with CSV  │    │ - Max 3 tries│
│   patterns  │    │   prompts    │    │             │    │              │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
       ▲                                                          │
       │                                                          │
       └──────────────────────────────────────────────────────────┘
                            Feedback Loop

Flow: CLI → Plan → Generate → Test → Self-Fix (≤3x) → Success
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# LangGraph and AI imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
import pandas as pd

# PDF processing
import pdfplumber
import PyPDF2

# AI providers
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentState:
    """State management for the agent workflow"""
    def __init__(self):
        self.target_bank: str = ""
        self.pdf_path: str = ""
        self.csv_path: str = ""
        self.pdf_content: str = ""
        self.csv_data: pd.DataFrame = None
        self.parser_code: str = ""
        self.test_results: Dict = {}
        self.attempts: int = 0
        self.max_attempts: int = 3
        self.error_feedback: str = ""
        self.success: bool = False


class BankStatementAgent:
    """Main agent class implementing the plan-generate-test-fix loop"""
    
    def __init__(self, api_provider: str = "gemini"):
        self.api_provider = api_provider
        self.llm_client = self._setup_llm()
        self.state = AgentState()
        self.workflow = self._build_workflow()
    
    def _setup_llm(self):
        """Initialize LLM client based on provider"""
        if self.api_provider == "gemini" and genai:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-1.5-flash')
        
        elif self.api_provider == "groq" and Groq:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment")
            return Groq(api_key=api_key)
        
        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("planner", self.plan_parsing_strategy)
        workflow.add_node("code_generator", self.generate_parser_code)
        workflow.add_node("tester", self.test_parser)
        workflow.add_node("evaluator", self.evaluate_and_decide)
        
        # Add edges
        workflow.add_edge("planner", "code_generator")
        workflow.add_edge("code_generator", "tester")
        workflow.add_edge("tester", "evaluator")
        
        # Conditional edges for self-correction
        workflow.add_conditional_edges(
            "evaluator",
            self._should_retry,
            {
                "retry": "code_generator",
                "success": END,
                "max_attempts": END
            }
        )
        
        workflow.set_entry_point("planner")
        return workflow.compile()
    
    def plan_parsing_strategy(self, state: Dict) -> Dict:
        """Analyze PDF and plan parsing strategy"""
        logger.info("🔍 Planning parsing strategy...")
        
        # Extract text from PDF
        pdf_content = self._extract_pdf_content(state["pdf_path"])
        
        # Analyze CSV structure
        csv_df = pd.read_csv(state["csv_path"])
        
        # Create planning prompt
        prompt = f"""
        Analyze this bank statement PDF content and plan a parsing strategy:
        
        PDF Content (first 2000 chars):
        {pdf_content[:2000]}
        
        Expected CSV Schema:
        Columns: {list(csv_df.columns)}
        Sample rows:
        {csv_df.head(3).to_string()}
        
        Bank: {state["target_bank"]}
        
        Please analyze:
        1. Document structure and layout
        2. Table patterns and positions
        3. Date/amount formats
        4. Transaction description patterns
        5. Column alignment and separators
        
        Provide a detailed parsing strategy in JSON format with:
        - extraction_method: (table_extraction, regex_patterns, coordinate_based)
        - date_format: detected format
        - amount_patterns: regex for amounts
        - table_structure: row/column analysis
        - parsing_challenges: potential issues
        """
        
        strategy = self._call_llm(prompt)
        
        state.update({
            "pdf_content": pdf_content,
            "csv_data": csv_df,
            "parsing_strategy": strategy
        })
        
        return state
    
    def generate_parser_code(self, state: Dict) -> Dict:
        """Generate parser code based on strategy"""
        logger.info("⚡ Generating parser code...")
        
        feedback = state.get("error_feedback", "")
        attempt_info = f" (Attempt {state['attempts'] + 1}/{self.state.max_attempts})"
        
        prompt = f"""
        Generate a Python parser for {state["target_bank"]} bank statements.
        
        Parsing Strategy: {state.get("parsing_strategy", "")}
        
        {f"Previous Error Feedback: {feedback}" if feedback else ""}
        
        Requirements:
        1. Create a function: parse(pdf_path: str) -> pd.DataFrame
        2. Return DataFrame with columns: {list(state["csv_data"].columns)}
        3. Use ONLY these imports: pandas as pd, pdfplumber, re, os
        4. Handle errors gracefully with try/except
        5. Extract transactions with proper data types
        6. Use pdfplumber.open(pdf_path) for PDF reading
        7. AVOID using undefined attributes like EmptyFileError or PLUMBER_ERRORS
        
        Expected output format matches:
        {state["csv_data"].head(2).to_string()}
        
        Generate complete, working Python code WITHOUT markdown formatting:
        
        import pandas as pd
        import pdfplumber
        import re
        import os
        
        def parse(pdf_path: str) -> pd.DataFrame:
            # Your parsing implementation here
            # Return DataFrame with exact columns: {list(state["csv_data"].columns)}
            pass
        """
        
        parser_code = self._call_llm(prompt)
        
        # Clean and format the code
        parser_code = self._clean_generated_code(parser_code)
        
        state.update({
            "parser_code": parser_code,
            "attempts": state.get("attempts", 0) + 1
        })
        
        return state
    
    def test_parser(self, state: Dict) -> Dict:
        """Test the generated parser"""
        logger.info("🧪 Testing generated parser...")
        
        # Write parser to file
        parser_path = f"custom_parsers/{state['target_bank']}_parser.py"
        with open(parser_path, 'w') as f:
            f.write(state["parser_code"])
        
        try:
            # Import and test the parser
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"{state['target_bank']}_parser", parser_path)
            parser_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(parser_module)
            
            # Check if parse function exists
            if not hasattr(parser_module, 'parse'):
                raise AttributeError("Parser module missing 'parse' function")
            
            # Run parser
            result_df = parser_module.parse(state["pdf_path"])
            
            # Basic validation
            if not isinstance(result_df, pd.DataFrame):
                raise TypeError(f"Parser returned {type(result_df)}, expected DataFrame")
            
            if result_df.empty:
                raise ValueError("Parser returned empty DataFrame")
            
            # Compare with expected CSV
            expected_df = state["csv_data"]
            
            # Test results
            tests_passed = self._compare_dataframes(result_df, expected_df)
            
            # Generate detailed error info if failed
            error_details = []
            if not tests_passed:
                error_details.append(f"Generated {result_df.shape[0]} rows, expected {expected_df.shape[0]}")
                error_details.append(f"Generated columns: {list(result_df.columns)}")
                error_details.append(f"Expected columns: {list(expected_df.columns)}")
                if not result_df.empty:
                    error_details.append(f"Sample generated data:\n{result_df.head(2).to_string()}")
            
            state.update({
                "test_results": {
                    "passed": tests_passed,
                    "generated_df": result_df,
                    "errors": error_details if not tests_passed else []
                }
            })
            
        except Exception as e:
            logger.error(f"Parser test failed: {str(e)}")
            state.update({
                "test_results": {
                    "passed": False,
                    "errors": [f"Parser execution error: {str(e)}"]
                }
            })
        
        return state
    
    def evaluate_and_decide(self, state: Dict) -> Dict:
        """Evaluate test results and decide next action"""
        logger.info("🎯 Evaluating results...")
        
        if state["test_results"]["passed"]:
            state["success"] = True
            logger.info("✅ Parser successfully generated and tested!")
        else:
            # Generate feedback for next attempt
            errors = state["test_results"]["errors"]
            feedback_prompt = f"""
            The generated parser failed with these errors:
            {errors}
            
            Provide specific feedback for fixing the parser:
            1. What went wrong?
            2. How to fix the extraction logic?
            3. What patterns or formats to adjust?
            
            Keep feedback concise and actionable.
            """
            
            feedback = self._call_llm(feedback_prompt)
            state["error_feedback"] = feedback
            
            logger.warning(f"❌ Attempt {state['attempts']} failed. Feedback: {feedback[:100]}...")
        
        return state
    
    def _should_retry(self, state: Dict) -> str:
        """Determine if should retry, succeed, or stop"""
        if state["test_results"]["passed"]:
            return "success"
        elif state["attempts"] >= self.state.max_attempts:
            return "max_attempts"
        else:
            return "retry"
    
    def _extract_pdf_content(self, pdf_path: str) -> str:
        """Extract text content from PDF"""
        try:
            # Try pdfplumber first
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
            try:
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed: {e2}")
                return f"Error extracting PDF: {e2}"
    
    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM"""
        try:
            if self.api_provider == "gemini":
                response = self.llm_client.generate_content(prompt)
                return response.text
            elif self.api_provider == "groq":
                response = self.llm_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="mixtral-8x7b-32768"
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error: LLM call failed"
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean and format generated code"""
        # Remove markdown code blocks
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        return code.strip()
    
    def _compare_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
        """Compare two DataFrames for equality"""
        try:
            # Basic shape check
            if df1.shape != df2.shape:
                logger.warning(f"Shape mismatch: {df1.shape} vs {df2.shape}")
                return False
            
            # Column names check
            if not all(df1.columns == df2.columns):
                logger.warning(f"Column mismatch: {list(df1.columns)} vs {list(df2.columns)}")
                return False
            
            # Check if we have data
            if df1.empty or df2.empty:
                logger.warning("One or both DataFrames are empty")
                return False
            
            # Content comparison with more lenient approach
            try:
                # Sort both DataFrames by the first column (Date) to ensure consistent ordering
                df1_sorted = df1.sort_values(by=df1.columns[0]).reset_index(drop=True)
                df2_sorted = df2.sort_values(by=df2.columns[0]).reset_index(drop=True)
                
                # Compare with some tolerance for floating point numbers
                return df1_sorted.equals(df2_sorted)
            except Exception as e:
                logger.warning(f"Direct comparison failed: {e}, trying alternative")
                # Alternative: check if at least 80% of rows match
                if len(df1) == len(df2):
                    matches = 0
                    for i in range(len(df1)):
                        if df1.iloc[i].equals(df2.iloc[i]):
                            matches += 1
                    similarity = matches / len(df1)
                    logger.info(f"Similarity: {similarity:.2%}")
                    return similarity >= 0.8
                return False
                
        except Exception as e:
            logger.error(f"DataFrame comparison error: {e}")
            return False
    
    def run(self, target_bank: str) -> bool:
        """Main execution method"""
        logger.info(f"🚀 Starting agent for {target_bank} bank...")
        
        # Setup paths
        data_dir = Path(f"data/{target_bank}")
        pdf_files = list(data_dir.glob("*.pdf"))
        csv_files = list(data_dir.glob("*.csv"))
        
        if not pdf_files or not csv_files:
            logger.error(f"Missing PDF or CSV files in {data_dir}")
            return False
        
        # Initialize state
        initial_state = {
            "target_bank": target_bank,
            "pdf_path": str(pdf_files[0]),
            "csv_path": str(csv_files[0]),
            "attempts": 0
        }
        
        # Run workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            return final_state.get("success", False)
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return False


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="AI Agent for Bank Statement Parsing")
    parser.add_argument("--target", required=True, help="Target bank name (e.g., icici)")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "groq"], 
                       help="LLM provider to use")
    
    args = parser.parse_args()
    
    # Initialize and run agent
    agent = BankStatementAgent(api_provider=args.provider)
    success = agent.run(args.target)
    
    if success:
        print(f"✅ Successfully generated parser for {args.target}")
        sys.exit(0)
    else:
        print(f"❌ Failed to generate parser for {args.target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
