"""
Test suite for bank statement parsers
"""
import pytest
import pandas as pd
import os
from pathlib import Path
import sys
import importlib.util


class TestBankParsers:
    """Test cases for generated bank parsers"""
    
    def test_icici_parser_exists(self):
        """Test that ICICI parser file was created"""
        parser_path = "custom_parsers/icici_parser.py"
        assert os.path.exists(parser_path), f"Parser file {parser_path} not found"
    
    def test_icici_parser_function(self):
        """Test that ICICI parser has correct function signature"""
        parser_path = "custom_parsers/icici_parser.py"
        
        # Import the parser module
        spec = importlib.util.spec_from_file_location("icici_parser", parser_path)
        parser_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser_module)
        
        # Check if parse function exists
        assert hasattr(parser_module, 'parse'), "parse function not found in parser"
        
        # Check function is callable
        assert callable(parser_module.parse), "parse is not callable"
    
    def test_icici_parser_output(self):
        """Test that ICICI parser produces correct output"""
        parser_path = "custom_parsers/icici_parser.py"
        
        # Import the parser
        spec = importlib.util.spec_from_file_location("icici_parser", parser_path)
        parser_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser_module)
        
        # Run parser on sample PDF
        pdf_path = "data/icici/icici sample.pdf"
        result_df = parser_module.parse(pdf_path)
        
        # Load expected CSV
        expected_df = pd.read_csv("data/icici/result.csv")
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Parser should return a DataFrame"
        assert not result_df.empty, "Parser should return non-empty DataFrame"
        
        # Check columns match
        assert list(result_df.columns) == list(expected_df.columns), \
            f"Columns mismatch. Expected: {list(expected_df.columns)}, Got: {list(result_df.columns)}"
        
        # Check shape matches
        assert result_df.shape == expected_df.shape, \
            f"Shape mismatch. Expected: {expected_df.shape}, Got: {result_df.shape}"
        
        # Check content equality (the main test)
        pd.testing.assert_frame_equal(
            result_df.sort_values(by=result_df.columns[0]).reset_index(drop=True),
            expected_df.sort_values(by=expected_df.columns[0]).reset_index(drop=True),
            check_dtype=False,  # Allow type differences
            check_exact=False   # Allow minor floating point differences
        )
    
    def test_parser_contract(self):
        """Test that parser follows the contract: parse(pdf_path) -> pd.DataFrame"""
        parser_path = "custom_parsers/icici_parser.py"
        
        spec = importlib.util.spec_from_file_location("icici_parser", parser_path)
        parser_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser_module)
        
        # Test with actual file
        pdf_path = "data/icici/icici sample.pdf"
        result = parser_module.parse(pdf_path)
        
        # Contract verification
        assert isinstance(result, pd.DataFrame), "parse() must return a pandas DataFrame"
        assert len(result.columns) > 0, "DataFrame must have columns"
        assert len(result) > 0, "DataFrame must have rows"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
