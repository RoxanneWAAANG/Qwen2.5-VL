#!/usr/bin/env python3
"""
Fix empty conversation turns and clean JSONL files
- Fix empty 'value' fields in conversations
- Keep only 'image' and 'conversations' fields
"""
import json
import argparse
from typing import Dict, Any, List

def fix_empty_turns(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix empty conversation turns"""
    fixed_conversations = []
    
    for i, conv in enumerate(conversations):
        if not isinstance(conv, dict):
            continue
            
        # Check for empty value field
        if conv.get('from') == 'gpt' and conv.get('value', '').strip() == '':
            # Generate a reasonable response based on context
            thoughts = conv.get('thoughts', '')
            actions = conv.get('actions', [])
            
            if actions:
                # If there are actions, this is a tool call response
                tool_name = actions[0].get('API_name', 'the tool') if actions else 'the tool'
                fixed_value = f"I'll use {tool_name} to analyze this further."
            elif 'analysis' in thoughts.lower() or 'results' in thoughts.lower():
                # If thoughts mention analysis or results, provide analysis response
                fixed_value = "Based on the analysis results, I can provide you with detailed information about the image."
            elif 'answer' in thoughts.lower():
                # If thoughts mention answering
                fixed_value = "Let me provide a comprehensive answer based on the available information."
            else:
                # Generic fallback
                fixed_value = "I understand your request and will proceed with the analysis."
            
            # Create fixed conversation turn
            fixed_conv = conv.copy()
            fixed_conv['value'] = fixed_value
            fixed_conversations.append(fixed_conv)
            
            print(f"    Fixed empty turn {i+1}: '{fixed_value}'")
        else:
            # Keep original turn
            fixed_conversations.append(conv)
    
    return fixed_conversations

def clean_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only 'image' and 'conversations' fields"""
    cleaned_sample = {
        'image': sample.get('image'),
        'conversations': sample.get('conversations', [])
    }
    return cleaned_sample

def process_file(input_file: str, output_file: str) -> Dict[str, int]:
    """Process file to fix empty turns and clean fields"""
    print(f"Processing {input_file} -> {output_file}")
    print("  - Fixing empty conversation turns")
    print("  - Removing unnecessary fields (keeping only 'image' and 'conversations')")
    
    stats = {
        'total_samples': 0,
        'samples_with_empty_turns': 0,
        'empty_turns_fixed': 0,
        'valid_samples': 0,
        'fields_cleaned': 0
    }
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                stats['total_samples'] += 1
                
                conversations = sample.get('conversations', [])
                if not conversations:
                    continue
                
                # Check for empty turns
                empty_turns = 0
                for conv in conversations:
                    if isinstance(conv, dict) and conv.get('value', '').strip() == '':
                        empty_turns += 1
                
                if empty_turns > 0:
                    stats['samples_with_empty_turns'] += 1
                    stats['empty_turns_fixed'] += empty_turns
                    print(f"  Line {line_num}: Fixing {empty_turns} empty turns")
                    
                    # Fix empty turns
                    fixed_conversations = fix_empty_turns(conversations)
                    sample['conversations'] = fixed_conversations
                
                # Clean sample (keep only required fields)
                cleaned_sample = clean_sample(sample)
                
                # Count cleaned fields
                original_fields = len(sample)
                cleaned_fields = len(cleaned_sample)
                if original_fields > cleaned_fields:
                    stats['fields_cleaned'] += 1
                
                # Write sample
                stats['valid_samples'] += 1
                outfile.write(json.dumps(cleaned_sample, ensure_ascii=False, separators=(',', ':')) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"JSON error at line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error at line {line_num}: {e}")
                continue
            
            if line_num % 10000 == 0:
                print(f"  Processed {line_num:,} lines...")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Fix empty conversation turns and clean JSONL files')
    parser.add_argument('input_file', help='Input JSONL file')
    parser.add_argument('output_file', help='Output cleaned JSONL file')
    
    args = parser.parse_args()
    
    stats = process_file(args.input_file, args.output_file)
    
    print(f"\n📊 Processing Results:")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Samples with empty turns: {stats['samples_with_empty_turns']:,}")
    print(f"  Empty turns fixed: {stats['empty_turns_fixed']:,}")
    print(f"  Samples with extra fields cleaned: {stats['fields_cleaned']:,}")
    print(f"  Valid samples output: {stats['valid_samples']:,}")
    
    improvements = []
    if stats['empty_turns_fixed'] > 0:
        improvements.append(f"Fixed {stats['empty_turns_fixed']} empty conversation turns")
    if stats['fields_cleaned'] > 0:
        improvements.append(f"Cleaned unnecessary fields from {stats['fields_cleaned']} samples")
    
    if improvements:
        print(f"\n✅ Successfully completed:")
        for improvement in improvements:
            print(f"  - {improvement}")
    else:
        print(f"\n✅ No issues found - file was already clean!")
    
    print(f"\n📝 Output file contains only 'image' and 'conversations' fields")
    print(f"   Removed fields: session_id, image_id, file_name, etc.")

if __name__ == "__main__":
    main()