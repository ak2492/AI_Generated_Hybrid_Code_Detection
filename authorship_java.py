import re
import numpy as np
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser(JAVA_LANGUAGE)

try:
    import nltk
    nltk.download('words', quiet=True)
    ENGLISH_WORDS = set(nltk.corpus.words.words())
except:
    ENGLISH_WORDS = set()

def extract_java_authorship(code):
    features = np.zeros(38)
    if not code or not str(code).strip(): return features
    
    lines = code.split('\n')
    total_lines = len(lines)
    chars = len(code)
    
    features[10] = np.mean([len(l) for l in lines]) if lines else 0 
    features[11] = max([len(l) for l in lines]) if lines else 0 
    features[12] = sum(1 for l in lines if not l.strip()) / max(total_lines, 1) 
    features[13] = code.count(' ') / max(chars, 1) 
    features[14] = 1 if code.startswith('    ') else 0 
    features[15] = 1 if ') {' in code else 0 
    
    comments = re.findall(r'//.*', code)
    multi_comments = re.findall(r'/\*[\s\S]*?\*/', code)
    features[16] = len(comments) / max(total_lines, 1)
    features[17] = len(multi_comments) / max(total_lines, 1) 
    features[18] = (len(comments) + len(multi_comments)) / max(total_lines, 1) 
    features[19] = np.mean([len(c) for c in comments]) if comments else 0 
    
    tree = parser.parse(bytes(code, "utf8"))
    
    node_count, max_depth, loops, conditionals, functions, classes = 0, 0, 0, 0, 0, 0
    cyclomatic, max_nest, lambdas, annotations, oo_patterns = 0, 0, 0, 0, 0
    switch_match, try_catch, returns, breaks, params, imports = 0, 0, 0, 0, 0, 0
    
    var_names, func_names, string_lits = [], [], 0

    def traverse(node, depth):
        nonlocal node_count, max_depth, loops, conditionals, functions, classes
        nonlocal cyclomatic, max_nest, lambdas, annotations, oo_patterns
        nonlocal switch_match, try_catch, returns, breaks, params, imports, string_lits
        
        node_count += 1
        max_depth = max(max_depth, depth)
        ntype = node.type
        
        if ntype in ['for_statement', 'enhanced_for_statement', 'while_statement', 'do_statement']: loops += 1; cyclomatic += 1
        elif ntype in ['if_statement']: conditionals += 1; cyclomatic += 1
        elif ntype == 'switch_statement': switch_match += 1; cyclomatic += 1
        elif ntype in ['method_declaration', 'constructor_declaration']: functions += 1
        elif ntype in ['class_declaration']: classes += 1
        elif ntype in ['interface_declaration', 'generic_type']: oo_patterns += 1
        elif ntype == 'try_statement': try_catch += 1
        elif ntype == 'lambda_expression': lambdas += 1
        elif ntype in ['marker_annotation', 'annotation']: annotations += 1
        elif ntype == 'return_statement': returns += 1
        elif ntype in ['break_statement', 'continue_statement']: breaks += 1
        elif ntype == 'formal_parameters': params += len(node.children)
        elif ntype == 'import_declaration': imports += 1
        elif ntype == 'string_literal': string_lits += (node.end_byte - node.start_byte)
        elif ntype == 'identifier':
            if node.parent and node.parent.type in ['variable_declarator', 'formal_parameter']:
                var_names.append(code[node.start_byte:node.end_byte])
            elif node.parent and node.parent.type in ['method_declaration']:
                func_names.append(code[node.start_byte:node.end_byte])
        
        if ntype == 'block': max_nest = max(max_nest, depth // 2)
        for child in node.children: traverse(child, depth + 1)

    traverse(tree.root_node, 0)
    
    features[0] = np.mean([len(n) for n in var_names]) if var_names else 0
    features[1] = np.mean([len(n) for n in func_names]) if func_names else 0
    
    all_names = var_names + func_names
    if all_names:
        features[2] = sum(1 for n in all_names if re.match(r'^[a-z]+[A-Z][a-zA-Z]*$', n)) / len(all_names)
        features[3] = sum(1 for n in all_names if '_' in n) / len(all_names)
        features[4] = sum(1 for n in all_names if n.isupper()) / len(all_names)
        features[5] = sum(1 for n in all_names if any(c.isdigit() for c in n)) / len(all_names)
        if ENGLISH_WORDS:
            features[36] = sum(1 for n in all_names if n.lower() in ENGLISH_WORDS) / len(all_names)
    
    words = re.findall(r'[a-zA-Z_]\w*', code)
    if words:
        keywords = {"public", "private", "class", "void", "static", "if", "else", "for", "while", "return"}
        features[6] = sum(1 for w in words if w in keywords) / len(words)
        features[7] = len(set(words)) / len(words)
        features[8] = np.mean([len(w) for w in words])
        
    features[9] = string_lits / max(chars, 1)
    features[20:36] = [node_count, max_depth, loops, conditionals, functions, classes, cyclomatic, max_nest, lambdas, annotations, oo_patterns, switch_match, try_catch, returns, breaks, params]
    features[37] = chars
    
    return features
