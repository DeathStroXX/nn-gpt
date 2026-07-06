import sys

path = "/shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evolver.py"
with open(path, "r") as f:
    code = f.read()

# 1. BASE_PROMPT_TEMPLATE
code = code.replace("4. Output EXACTLY ONE method: `{method_name}`.", "4. Output the exact code for the requested method(s): `{method_names}`.")
code = code.replace("You must rewrite the following component: `{method_name}`", "You must rewrite the following component(s): `{method_names}`")

# 2. evolve_component signature & top logic
old_evolve_top = """    def evolve_component(self, method_name, attempt=1, total_attempts=5):
        print(f"\\n[Meta] Evolving: {method_name}")
        with open(TARGET_FILE, 'r') as f: full_code = f.read()
        
        orig_code, span, indent_col = self._extract_method(full_code, method_name)
        if not orig_code: return"""

new_evolve_top = """    def evolve_component(self, method_names, attempt=1, total_attempts=5):
        if isinstance(method_names, str):
            method_names = [method_names]
        methods_str = ", ".join(method_names)
        
        print(f"\\n[Meta] Evolving: {methods_str}")
        with open(TARGET_FILE, 'r') as f: full_code = f.read()
        
        orig_codes = []
        spans = []
        indent_cols = []
        for name in method_names:
            c, span, indent = self._extract_method(full_code, name)
            if not c:
                print(f"[Meta] Failed to extract {name} from target code.")
                return False
            orig_codes.append(f"### {name} ###\\n{c}")
            spans.append(span)
            indent_cols.append(indent)
        
        orig_code = "\\n\\n".join(orig_codes)"""
code = code.replace(old_evolve_top, new_evolve_top)

# 3. Prompt format
old_prompt_format = """        prompt = BASE_PROMPT_TEMPLATE.format(
            search_space=SEARCH_SPACE_STR,
            full_code=skel_full_code,
            method_name=method_name,
            task_specific_instructions=INSTRUCTIONS.get(method_name, ""),"""

new_prompt_format = """        prompt = BASE_PROMPT_TEMPLATE.format(
            search_space=SEARCH_SPACE_STR,
            full_code=skel_full_code,
            method_names=methods_str,
            task_specific_instructions="\\n".join([INSTRUCTIONS.get(n, "") for n in method_names]),"""
code = code.replace(old_prompt_format, new_prompt_format)

# 4. Extraction & Injection logic
old_injection = """        # Cleanup & Extraction
        extracted = self._extract_function_body(raw_res, method_name)
        
        if extracted:
            new_code = extracted
        else:
            print(f"[Meta] Failed to extract valid {method_name} from LLM response. Skipping.")
            return

        # Normalize Indentation
        new_code = textwrap.dedent(new_code).strip()
        
        print(f"--- [DEBUG] Cleaned Code (Normalized) ---\\n{new_code}\\n---------------------------")
        
        # Indentation for Injection
        reindented = "\\n".join([" " * indent_col + line for line in new_code.splitlines()])

        # Syntax Check & Injection
        valid_syntax = False
        try:
            test_full = full_code[:span[0]] + reindented + "\\n" + full_code[span[1]:]

            # DEBUG: Print what we are trying to inject
            print(f"--- [DEBUG] Injection Snippet ---\\n{test_full[span[0]-50:span[0]+len(reindented)+50]}\\n-------------------------------")

            ast.parse(test_full)
            valid_syntax = True
        except SyntaxError as e:
            print(f"[Meta] Syntax Error: {e}")"""

new_injection = """        # Cleanup & Extraction
        extracted_codes = []
        for name in method_names:
            extracted = self._extract_function_body(raw_res, name)
            if not extracted:
                print(f"[Meta] Failed to extract valid {name} from LLM response. Skipping.")
                return False
            extracted_codes.append(extracted)

        # Sort spans backwards to preserve string indices during multiple replacements
        replacements = list(zip(spans, indent_cols, extracted_codes))
        replacements.sort(key=lambda x: x[0][0], reverse=True)

        valid_syntax = False
        try:
            test_full = full_code
            for span, indent_col, new_code in replacements:
                new_code = textwrap.dedent(new_code).strip()
                reindented = "\\n".join([" " * indent_col + line for line in new_code.splitlines()])
                test_full = test_full[:span[0]] + reindented + "\\n" + test_full[span[1]:]

            ast.parse(test_full)
            valid_syntax = True
        except SyntaxError as e:
            print(f"[Meta] Syntax Error: {e}")"""
code = code.replace(old_injection, new_injection)

# 5. Backup file name
old_bkp = """            bkp = os.path.join(BACKUP_DIR, f"{target_filename}_{method_name}.bak")"""
new_bkp = """            bkp = os.path.join(BACKUP_DIR, f"{target_filename}_{'_'.join(method_names)}.bak")"""
code = code.replace(old_bkp, new_bkp)

with open(path, "w") as f:
    f.write(code)

print("Patch applied successfully.")
