import os, re
import astor
import tqdm
import keyword
from ast import parse
import ast
import string
from ast import FunctionDef
from cognitive_complexity.api import get_cognitive_complexity
# from eval import *
from radon.complexity import cc_visit

def clean_code(code_str):
    cleaned_code = code_str.replace('\n\n', '\n')
    cleaned_code = '\n'.join(filter(lambda x: len(x.strip(' ')) > 0, cleaned_code.split('\n')))
    
    return cleaned_code


def remove_comments(code):
    pattern = r"(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\'|\".*?\"|'.*?'|#[^\n]*)"
    code_without_comments = re.sub(pattern, "", code, flags=re.DOTALL)
    return code_without_comments


def count_non_empty_lines(code_str):
    lines = code_str.split("\n")
    non_empty_lines = [line for line in lines if line.strip()]
    return len(non_empty_lines)


def remove_imports(code:str) -> str:
    # remove `import` lines from the code string
    lines = code.split('\n')
    lines = [line for line in lines if not line.startswith('import') and not (line.startswith('from') and 'import' in line)]
    return '\n'.join(lines)

def remove_decorators(code:str) -> str:
    # remove `@decorator` lines from the code string
    lines = code.split('\n')
    lines = [line for line in lines if not line.lstrip().startswith('@')]
    return '\n'.join(lines)


def reformat_indent(code:str)->str:
    lines = code.split('\n')  # Split the code into individual lines

    # Find the minimum indentation level
    min_indent = float('inf')
    for line in lines:
        if line.strip():  # Ignore empty lines
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)

    # Remove the minimum indentation from each line
    cleaned_code = ""
    for line in lines:
        cleaned_code += line[min_indent:] + '\n'

    return cleaned_code


def quantify_code_complexity(code):
    code = remove_imports(code)
    code = remove_decorators(code)
    code = reformat_indent(code)
    # code = code.replace('""" ', '"""\n')
    try:
        results = cc_visit(code)
    except Exception as e:
        raise e
    
        # print(e)
        # print(code)
    # in case of multiple functions, return the maximum complexity
    complexity = max([result.complexity for result in results])
    return complexity


def quantify_Cognitive_code_complexity(code_str):
    """
    Cognitive Complexity is a measure of how difficult a unit of code is to intuitively understand. 
    Unlike Cyclomatic Complexity, which determines how difficult your code will be to test, 
    Cognitive Complexity tells you how difficult your code will be to read and understand.

    Args:
        code_str (_type_): _description_
    """
    code_str = remove_imports(code_str)
    code_str = remove_decorators(code_str)
    code_str = reformat_indent(code_str)
    # code_str = code_str.replace('""" ', '"""\n')
    try:
        funcdef = ast.parse(code_str).body[0]
    except Exception as e:
        # print(e)
        # print(code_str)
        raise e
        
    cog_comp = get_cognitive_complexity(funcdef)
    
    return cog_comp


def remove_print_content(code_str:str) -> str:
    """
    Remove the content of print statements from the code string

    Args:
        code_str (str): The code string

    Returns:
        str: The code string with the content of print statements removed
    """
    # Define the regular expression pattern for print statements
    print_pattern = r'print\((.*)\)'

    # clean the code string by stripping
    code_str = code_str.strip()
    
    # Replace the content of print statements with an empty string
    code_str_without_print_content = re.sub(print_pattern, 'print()', code_str)

    return code_str_without_print_content

