import ast
import re
import numpy as np

class AuthorshipVisitor(ast.NodeVisitor):
    def __init__(self):
        self.node_count = self.max_depth = self.current_depth = 0
        self.loops = self.conditionals = self.functions = self.classes = self.list_comps = 0
        
    def generic_visit(self, node):
        self.node_count += 1
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        super().generic_visit(node)
        self.current_depth -= 1
        
    def visit_For(self, node): self.loops += 1; self.generic_visit(node)
    def visit_While(self, node): self.loops += 1; self.generic_visit(node)
    def visit_If(self, node): self.conditionals += 1; self.generic_visit(node)
    def visit_FunctionDef(self, node): self.functions += 1; self.generic_visit(node)
    def visit_ClassDef(self, node): self.classes += 1; self.generic_visit(node)
    def visit_ListComp(self, node): self.list_comps += 1; self.generic_visit(node)

def extract_authorship(code):
    features = np.zeros(38)
    lines = code.split('\n')
    total_lines = len(lines)
    features[0] = total_lines
    features[1] = sum(1 for line in lines if not line.strip()) / max(total_lines, 1)
    features[2] = max((len(line) for line in lines), default=0)
    features[3] = np.mean([len(line) for line in lines]) if lines else 0
    
    words = re.findall(r'[a-zA-Z_]\w*', code)
    features[4] = len(words)
    features[5] = len(set(words)) / max(len(words), 1)
    features[6] = sum(len(w) for w in words) / max(len(words), 1)
    features[7] = sum(1 for w in words if re.match(r'^[a-z]+[A-Z][a-zA-Z]*$', w))
    features[8] = sum(1 for w in words if '_' in w)
    features[9] = sum(1 for w in words if w.isupper())
    
    try:
        tree = ast.parse(code)
        visitor = AuthorshipVisitor()
        visitor.visit(tree)
        features[10:17] = [visitor.node_count, visitor.max_depth, visitor.loops, 
                           visitor.conditionals, visitor.functions, visitor.classes, visitor.list_comps]
    except SyntaxError:
        pass 
        
    return features
