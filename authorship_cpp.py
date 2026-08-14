import re
import numpy as np
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

# Initialize the Tree-sitter C++ parser
CPP_LANGUAGE = Language(tscpp.language())
parser = Parser(CPP_LANGUAGE)

def extract_cpp_authorship(code):
    features = np.zeros(38)
    
    # --- Layout Features ---
    lines = code.split('\n')
    total_lines = len(lines)
    features[0] = total_lines
    features[1] = sum(1 for line in lines if not line.strip()) / max(total_lines, 1)
    features[2] = max((len(line) for line in lines), default=0)
    features[3] = np.mean([len(line) for line in lines]) if lines else 0
    
    # --- Lexical Features ---
    words = re.findall(r'[a-zA-Z_]\w*', code)
    features[4] = len(words)
    features[5] = len(set(words)) / max(len(words), 1)
    features[6] = sum(len(w) for w in words) / max(len(words), 1)
    features[7] = sum(1 for w in words if re.match(r'^[a-z]+[A-Z][a-zA-Z]*$', w))
    features[8] = sum(1 for w in words if '_' in w)
    features[9] = sum(1 for w in words if w.isupper())
    
    # --- Syntactic Features (Tree-sitter AST) ---
    tree = parser.parse(bytes(code, "utf8"))
    
    node_count = 0
    max_depth = 0
    loops = 0
    conditionals = 0
    functions = 0
    classes = 0

    def traverse(node, depth):
        nonlocal node_count, max_depth, loops, conditionals, functions, classes
        node_count += 1
        max_depth = max(max_depth, depth)
        
        node_type = node.type
        if node_type in ['for_statement', 'while_statement', 'do_statement']:
            loops += 1
        elif node_type in ['if_statement', 'switch_statement']:
            conditionals += 1
        elif node_type == 'function_definition':
            functions += 1
        elif node_type in ['class_specifier', 'struct_specifier']:
            classes += 1
            
        for child in node.children:
            traverse(child, depth + 1)

    traverse(tree.root_node, 0)
    
    features[10:16] = [node_count, max_depth, loops, conditionals, functions, classes]
    
    return features
