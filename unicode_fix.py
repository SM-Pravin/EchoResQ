#!/usr/bin/env python3
"""
Unicode Compatibility Fix for Emergency AI
Replaces Unicode characters with ASCII-compatible alternatives for Windows compatibility.
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict


class UnicodeCompatibilityFixer:
    """Fix Unicode compatibility issues in Emergency AI files."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.working_files = self.project_root / "WORKING_FILES"
        
        # Unicode character replacements for Windows compatibility
        self.unicode_replacements = {
            '[EMERGENCY]': '[EMERGENCY]',
            '[CONFIG]': '[CONFIG]',
            '[DASHBOARD]': '[DASHBOARD]',
            '[WARNING]': '[WARNING]',
            '[OK]': '[OK]',
            '[ERROR]': '[ERROR]',
            '[STAR]': '[STAR]',
            '[TARGET]': '[TARGET]',
            '[COMPUTER]': '[COMPUTER]',
            '[SUCCESS]': '[SUCCESS]',
            '[DEBUG]': '[DEBUG]',
            '[INFO]': '[INFO]',
            '[SEARCH]': '[SEARCH]',
            '[CHART]': '[CHART]',
            '[STAR]': '[STAR]',
            '[ROCKET]': '[ROCKET]',
        }
    
    def fix_file(self, file_path: Path) -> bool:
        """Fix Unicode characters in a single file."""
        try:
            # Read file with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file contains any Unicode characters we need to replace
            has_unicode = any(char in content for char in self.unicode_replacements.keys())
            
            if not has_unicode:
                return True  # No changes needed
            
            # Apply replacements
            modified_content = content
            for unicode_char, replacement in self.unicode_replacements.items():
                modified_content = modified_content.replace(unicode_char, replacement)
            
            # Write back with UTF-8 BOM for better Windows compatibility
            with open(file_path, 'w', encoding='utf-8-sig') as f:
                f.write(modified_content)
            
            print(f"[OK] Fixed Unicode in: {file_path.relative_to(self.project_root)}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error fixing {file_path}: {e}")
            return False
    
    def fix_all_files(self) -> int:
        """Fix Unicode characters in all Python files."""
        print("[INFO] Fixing Unicode compatibility issues...")
        
        # Find all Python files
        python_files = []
        for pattern in ["**/*.py"]:
            python_files.extend(self.working_files.glob(pattern))
        
        # Also check root level files
        root_python_files = list(self.project_root.glob("*.py"))
        python_files.extend(root_python_files)
        
        fixed_count = 0
        total_files = len(python_files)
        
        for file_path in python_files:
            if file_path.name.startswith('.'):
                continue  # Skip hidden files
                
            if self.fix_file(file_path):
                fixed_count += 1
        
        print(f"\n[SUMMARY] Unicode Fix Summary:")
        print(f"  Total files processed: {total_files}")
        print(f"  Files successfully fixed: {fixed_count}")
        print(f"  Files with errors: {total_files - fixed_count}")
        
        return fixed_count
    
    def validate_fix(self) -> bool:
        """Validate that Unicode fixes work by testing imports."""
        print("\n[INFO] Validating Unicode fixes...")
        
        try:
            # Add WORKING_FILES to path
            sys.path.insert(0, str(self.working_files))
            
            # Test imports that were failing due to Unicode issues
            test_imports = [
                ("tests.stress_test_suite", "StressTestSuite"),
                ("cli", "main"),
                ("gui", "EmergencyAIGUI"),
                ("validate", "EmergencyAIValidator"),
            ]
            
            success_count = 0
            for module_name, class_or_func in test_imports:
                try:
                    module = __import__(module_name, fromlist=[class_or_func])
                    getattr(module, class_or_func)
                    print(f"[OK] {module_name}.{class_or_func} imports successfully")
                    success_count += 1
                except Exception as e:
                    print(f"[ERROR] {module_name}.{class_or_func} import failed: {e}")
            
            success_rate = (success_count / len(test_imports)) * 100
            print(f"\n[STATS] Import Success Rate: {success_rate:.1f}% ({success_count}/{len(test_imports)})")
            
            return success_count == len(test_imports)
            
        except Exception as e:
            print(f"[ERROR] Validation failed: {e}")
            return False


def main():
    """Main Unicode compatibility fix function."""
    print("Emergency AI - Unicode Compatibility Fix")
    print("=" * 50)
    
    fixer = UnicodeCompatibilityFixer()
    
    try:
        # Fix all files
        fixed_count = fixer.fix_all_files()
        
        if fixed_count > 0:
            print(f"\n[SUCCESS] Successfully fixed Unicode issues in {fixed_count} files")
        else:
            print("\n[INFO] No files needed Unicode fixes")
        
        # Validate the fixes
        validation_success = fixer.validate_fix()
        
        if validation_success:
            print("\n[SUCCESS] All Unicode compatibility issues resolved!")
            print("Emergency AI should now work properly on Windows systems.")
            return 0
        else:
            print("\n[WARNING] Some import issues remain after Unicode fixes")
            print("Additional debugging may be required.")
            return 1
            
    except Exception as e:
        print(f"\n[ERROR] Unicode fix failed: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())