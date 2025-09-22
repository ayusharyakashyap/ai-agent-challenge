import pandas as pd
import pdfplumber
import re


def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parse ICICI bank statement PDF and return structured data
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        DataFrame with columns: Date, Description, Debit Amt, Credit Amt, Balance
    """
    transactions = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                    
                lines = text.split('\n')
                
                for line in lines:
                    # Skip header lines and bank name
                    if any(x in line.lower() for x in ['date description', 'debit amt', 'credit amt', 'balance', 'karbon', 'bannk']):
                        continue
                    
                    # Look for date pattern (DD-MM-YYYY)
                    date_pattern = r'(\d{2}-\d{2}-\d{4})'
                    date_match = re.search(date_pattern, line)
                    
                    if date_match:
                        date = date_match.group(1)
                        
                        # Remove date from line to parse the rest
                        remaining = line[date_match.end():].strip()
                        
                        # Find all decimal numbers in the line (amounts and balance)
                        numbers = re.findall(r'\d+\.?\d*', remaining)
                        
                        if len(numbers) >= 2:
                            # The last number is always the balance
                            balance = numbers[-1]
                            
                            # Extract description (everything before the first number)
                            # Split by the first number found
                            first_number_match = re.search(r'\d+\.?\d*', remaining)
                            if first_number_match:
                                description = remaining[:first_number_match.start()].strip()
                                
                                # Determine if it's debit or credit based on description keywords
                                credit_keywords = ['salary', 'credit', 'deposit', 'interest', 'transfer from']
                                debit_keywords = ['payment', 'purchase', 'withdrawal', 'charge', 'debit', 'emi', 'bill']
                                
                                desc_lower = description.lower()
                                
                                if any(keyword in desc_lower for keyword in credit_keywords):
                                    # It's a credit transaction
                                    debit_amt = ''
                                    credit_amt = numbers[0] if numbers else ''
                                else:
                                    # It's a debit transaction (default)
                                    debit_amt = numbers[0] if numbers else ''
                                    credit_amt = ''
                                
                                # Add to transactions if we have a valid description
                                if description and len(description) > 3:
                                    transactions.append({
                                        'Date': date,
                                        'Description': description,
                                        'Debit Amt': debit_amt,
                                        'Credit Amt': credit_amt,
                                        'Balance': balance
                                    })
    
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    return df


# Test function when run directly
if __name__ == "__main__":
    result = parse('data/icici/icici sample.pdf')
    print(f"Parsed {len(result)} transactions")
    print(result.head())
