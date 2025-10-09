# Instruction Dataset Building

## Table of Contents
- [Overview](#overview)
- [Dataset Types](#dataset-types)
- [Single-Round Dialogue](#single-round-dialogue)
- [One-Tool Multi-Round Dialogue](#one-tool-multi-round-dialogue)
- [Multi-Tool Multi-Round Dialogue](#multi-tool-multi-round-dialogue)
- [Image Handling Scenarios](#image-handling-scenarios)
- [Implementation Guidelines](#implementation-guidelines)
- [Quality Assurance](#quality-assurance)

## Overview

Our instruction dataset building pipeline supports three levels of complexity:

1. **Single-Round**: Basic tool usage (foundation datasets)
2. **One-Tool Multi-Round**: Iterative refinement with same tool
3. **Multi-Tool Multi-Round**: Complex workflows across different tools

Each level builds upon the previous, enabling progressive training from basic tool usage to complex reasoning chains.

## Dataset Types

### Foundation Datasets (Single-Round)
- `unigradicon_reg_dataset.jsonl` - Image registration
- `ultrasam_seg_dataset.jsonl` - Image segmentation  
- `healthgpt_superres_dataset.jsonl` - Image super-resolution
- `internet_seg_dataset.jsonl` - Fundus segmentation
- `llava_rad_rg_dataset.jsonl` - Radiology report generation
- `llava_sum_dataset.jsonl` - Text summarization
- `rate_ner_dataset.jsonl` - Named entity recognition
- `pmc_llama_medqa_dataset.jsonl` - Medical question answering
- `svlms_fundus_dataset.jsonl` - Specialist analysis

### Generated Datasets (Multi-Round)
- `one_tool_multiround.jsonl` - Single tool refinement conversations
- `multi_tool_multiround.jsonl` - Cross-tool workflow conversations

## Single-Round Dialogue

### Description
Basic tool usage where user makes one request and assistant provides one complete response using a single tool.

### Structure
```
User Request → Assistant Tool Call → Tool Output → Assistant Final Response
```

### Example Format
```json
{
  "session_id": "single_001",
  "image_id": "sample_001", 
  "image": "/path/to/image.jpg",
  "file_name": "/path/to/image.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nSegment the lung regions in this chest X-ray"
    },
    {
      "from": "gpt", 
      "thoughts": "I need to use UltraSAM to segment the lung regions in this chest X-ray image.",
      "actions": [{"API_name": "UltraSAM", "API_params": {"prompt": "lung"}}],
      "value": "I'll use UltraSAM to segment the lung regions in your chest X-ray."
    },
    {
      "from": "human",
      "value": "UltraSAM output: Segmentation complete. Lung regions identified with 95% accuracy.\n\nAnswer my first request: Segment the lung regions in this chest X-ray"
    },
    {
      "from": "gpt",
      "thoughts": "Based on the UltraSAM output, I can provide the segmentation results.",
      "actions": [],
      "value": "The lung segmentation has been completed successfully. The UltraSAM model has identified the lung regions with 95% accuracy, providing precise boundaries for both left and right lungs."
    }
  ]
}
```

### Use Cases
- **Tool validation**: Ensure each tool works correctly
- **Basic training**: Teach models fundamental tool usage
- **Data quality**: High-precision examples for each tool
- **Foundation building**: Base examples for multi-round generation

## One-Tool Multi-Round Dialogue

### Description
Iterative refinement using the same tool multiple times with parameter adjustments, clarifications, or progressive enhancement.

### Structure
```
User Request → Tool Call → Refinement Request → Tool Call → Final Request → Tool Call
```

### Scenario Types

#### Scenario A: Parameter Refinement
**Use Case**: User requests parameter adjustments for better results

```json
{
  "session_id": "one_tool_001",
  "image_id": "sample_002",
  "image": "/path/to/ct_scan.jpg", 
  "file_name": "/path/to/ct_scan.jpg",
  "conversations": [
    {
      "from": "human", 
      "value": "<image>\nSegment the liver in this CT scan"
    },
    {
      "from": "gpt",
      "thoughts": "I'll use UltraSAM to segment the liver region.",
      "actions": [{"API_name": "UltraSAM", "API_params": {"prompt": "liver"}}],
      "value": "I'll segment the liver region using UltraSAM."
    },
    {
      "from": "human",
      "value": "UltraSAM output: Liver segmentation complete but includes some adjacent tissues.\n\nThe segmentation is too broad. Can you make it more precise?"
    },
    {
      "from": "gpt",
      "thoughts": "I need to refine the segmentation with more specific parameters.",
      "actions": [{"API_name": "UltraSAM", "API_params": {"prompt": "liver", "precision": "high"}}],
      "value": "I'll refine the segmentation with higher precision to focus specifically on liver tissue."
    },
    {
      "from": "human", 
      "value": "UltraSAM output: Refined liver segmentation complete with improved boundary precision.\n\nPerfect! Can you also highlight any abnormal areas?"
    },
    {
      "from": "gpt",
      "thoughts": "Now I'll run another segmentation focusing on abnormalities within the liver.",
      "actions": [{"API_name": "UltraSAM", "API_params": {"prompt": "liver abnormalities", "highlight": true}}],
      "value": "I'll now run a focused analysis to highlight any abnormal areas within the segmented liver region."
    }
  ]
}
```

#### Scenario B: Progressive Enhancement
**Use Case**: User builds complexity gradually

```json
{
  "conversations": [
    {"from": "human", "value": "<image>\nWhat's in this medical image?"},
    {"from": "gpt", "value": "Basic analysis..."},
    {"from": "human", "value": "Can you provide more technical details?"},
    {"from": "gpt", "value": "Detailed technical analysis..."},
    {"from": "human", "value": "Now generate a clinical report format"},
    {"from": "gpt", "value": "Formal clinical report..."}
  ]
}
```

### Benefits
- **Tool mastery**: Deep understanding of individual tool capabilities
- **Parameter optimization**: Learning optimal settings for different scenarios
- **Iterative refinement**: Teaching progressive improvement strategies
- **User interaction**: Natural back-and-forth conversation patterns

## Multi-Tool Multi-Round Dialogue

### Description
Complex workflows that chain multiple tools together, simulating real medical analysis pipelines.

### Structure
```
Image Input → Tool A → Tool B → Tool C → Comprehensive Analysis
```

### Medical Workflow Chains

#### Chain 1: Complete Imaging Pipeline
```
Registration → Segmentation → Report Generation → Summarization → QA
```

#### Chain 2: Diagnostic Workflow  
```
Image Enhancement → Analysis → Specialist Review → Entity Extraction → Documentation
```

#### Chain 3: Comparative Analysis
```
Image A Analysis → Image B Analysis → Registration → Comparison Report → Clinical Summary
```

## Image Handling Scenarios

### Scenario 1: Single Image Throughout Conversation

**Use Case**: Sequential analysis of one image with multiple tools

**Example Workflow**:
```
Turn 1: Segmentation of lung regions
Turn 2: Report generation based on segmentation  
Turn 3: Text summarization of report
Turn 4: Entity extraction from summary
```

**Output Format**:
```json
{
  "session_id": "multi_001",
  "image_id": "chest_xray_001",
  "image": "/path/to/chest_xray.jpg",
  "file_name": "/path/to/chest_xray.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nSegment the lung regions in this chest X-ray"},
    {"from": "gpt", "thoughts": "...", "actions": [...], "value": "..."},
    {"from": "human", "value": "UltraSAM output: ..."},
    {"from": "gpt", "thoughts": "...", "actions": [], "value": "..."},
    {"from": "human", "value": "Following up on the previous analysis, generate a radiology report"},
    {"from": "gpt", "thoughts": "...", "actions": [...], "value": "..."}
  ]
}
```

### Scenario 2: Registration with Two Images from Start

**Use Case**: Comparative analysis requiring image alignment

**Example Workflow**:
```
Turn 1: Register pre/post treatment scans
Turn 2: Segment regions of interest
Turn 3: Generate comparison report
```

**Output Format**:
```json
{
  "session_id": "multi_002", 
  "image_id": "ct_pre_001",
  "image": ["/path/to/ct_pre.jpg", "/path/to/ct_post.jpg"],
  "file_name": "/path/to/ct_pre.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\n<image>\nRegister these pre and post-treatment CT scans"},
    {"from": "gpt", "thoughts": "...", "actions": [...], "value": "..."}
  ]
}
```

### Scenario 3: Registration - Add Second Image Later

**Use Case**: User starts with single image analysis, then adds comparison image

**Example Workflow**:
```
Turn 1: Analyze baseline scan
Turn 2: Generate initial report
Turn 3: Add follow-up scan for registration
Turn 4: Generate comparative analysis
```

**Output Format**:
```json
{
  "session_id": "multi_003",
  "image_id": "baseline_001", 
  "image": ["/path/to/baseline.jpg", "/path/to/baseline_ref.jpg"],
  "file_name": "/path/to/baseline.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nAnalyze this baseline MRI scan"},
    {"from": "gpt", "value": "Initial analysis..."},
    {"from": "human", "value": "Now I have a second image. <image>\nRegister this follow-up scan"},
    {"from": "gpt", "value": "Registration analysis..."}
  ]
}
```

### Scenario 4: Switch to Different Image Mid-Conversation

**Use Case**: User analyzes multiple unrelated images in same conversation

**Example Workflow**:
```
Turn 1-2: Analyze chest X-ray
Turn 3-4: Switch to brain MRI analysis
Turn 5: Compare findings
```

**Output Format**:
```json
{
  "session_id": "multi_004",
  "image_id": "chest_001",
  "image": ["/path/to/chest_xray.jpg", "/path/to/brain_mri.jpg"], 
  "file_name": "/path/to/chest_xray.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nAnalyze this chest X-ray"},
    {"from": "gpt", "value": "Chest analysis..."},
    {"from": "human", "value": "Now I have a different image to analyze. <image>\nExamine this brain MRI"},
    {"from": "gpt", "value": "Brain MRI analysis..."}
  ]
}
```

### Scenario 5: Mixed Image and Text-Only Tools

**Use Case**: Complete workflow including text processing

**Example Workflow**:
```
Turn 1: Image analysis
Turn 2: Report generation  
Turn 3: Named entity recognition (text-only)
Turn 4: Question answering (text-only)
```

**Output Format**:
```json
{
  "session_id": "multi_005",
  "image_id": "pathology_001",
  "image": "/path/to/pathology_slide.jpg",
  "file_name": "/path/to/pathology_slide.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nAnalyze this pathology slide"},
    {"from": "gpt", "value": "Pathology analysis..."},
    {"from": "human", "value": "Now, extract medical entities from the report"},
    {"from": "gpt", "value": "Entity extraction without image tags..."}
  ]
}
```

## Implementation Guidelines

### Tool Chain Design Principles

#### Logical Workflow Chains
```python
workflow_chains = {
    "UniGradICON": ["UltraSAM", "LLaVA-Rad", "HealthGPT"],  # Registration → Analysis
    "UltraSAM": ["LLaVA-Rad", "SpecialistVLMs"],           # Segmentation → Report  
    "HealthGPT": ["LLaVA-Rad", "UltraSAM"],                # Enhancement → Analysis
    "IterNet": ["SpecialistVLMs", "LLaVA-Rad"],            # Fundus → Specialist
    "LLaVA-Rad": ["LLaVA", "PMC-LLaMA"],                   # Report → Summary/QA
    "LLaVA": ["RaTE-NER", "PMC-LLaMA"],                    # Summary → NER/QA
    "RaTE-NER": ["PMC-LLaMA"],                             # NER → QA
    "PMC-LLaMA": ["LLaVA"],                                # QA → Summary
    "SpecialistVLMs": ["LLaVA", "PMC-LLaMA"]               # Specialist → Summary/QA
}
```

#### Diversity Strategy
- **Short chains** (2-3 tools): 40% probability
- **Medium chains** (3-5 tools): 40% probability  
- **Long chains** (5+ tools): 20% probability

#### Context Awareness
- **Same image**: `"Following up on the previous analysis..."`
- **Add registration**: `"Now I have a second image. <image>..."`
- **Switch image**: `"Now I have a different image to analyze. <image>..."`
- **Text continuation**: `"Using the previous output..."`

### Image Management Rules

#### Image Field Logic
```python
# Single image
"image": "/path/to/single.jpg"

# Multiple images  
"image": ["/path/to/image1.jpg", "/path/to/image2.jpg"]
```

#### File Name Consistency
- Always points to **primary/first** image
- Maintains metadata consistency across conversation
- Used for dataset organization and retrieval

#### Tool-Specific Requirements
```python
image_requirements = {
    "UniGradICON": {"min_images": 2, "max_images": 2, "requires_pair": True},
    "UltraSAM": {"min_images": 1, "max_images": 1, "requires_pair": False},
    "RaTE-NER": {"min_images": 0, "max_images": 0, "requires_pair": False},
    "PMC-LLaMA": {"min_images": 0, "max_images": 1, "requires_pair": False}
}
```

### Generation Parameters

#### Probability Controls
- **New image introduction**: 30% chance after turn 2
- **Registration addition**: Triggered by UniGradICON tool
- **Text-only continuation**: Based on tool requirements

#### Quality Thresholds
- **Minimum conversation length**: 2 turns (4 messages)
- **Maximum conversation length**: 6 turns (12 messages) 
- **Tool availability check**: Ensure real data exists
- **Format validation**: Verify Qwen template compliance

## Quality Assurance

### Performance Metrics

#### Coverage Metrics
- **Tool usage distribution**: Balanced across all 9 tools
- **Chain length distribution**: Appropriate mix of short/medium/long
- **Image scenario coverage**: All 5 scenarios represented
- **Workflow complexity**: Progressive difficulty levels

#### Quality Metrics
- **Conversation coherence**: Context maintained across turns
- **Tool appropriateness**: Logical tool selection
- **Response relevance**: On-topic assistant responses
- **Format consistency**: Uniform structure across dataset

