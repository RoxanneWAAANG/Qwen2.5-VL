# Instruction Dataset Building

## Dataset Source

| Dataset                         | Samples |
|---------------------------------|--------:|
| BiomedClip                      | 8,804   |
| CellSAM                         | 2,000   |
| CellViT                         | 2,000   |
| ChatCAD-G                       | 4,980   |
| ChatCAD-R                       | 2,814   |
| CONCH                           | 4,000   |
| DSMIL                           | 2,000   |
| Grounding Dino + MedSAM         | 1,999   |
| Grounding Dino                  | 4,759   |
| HealthGPT -- Reconstruction     | 5,000   |
| HealthGPT -- Super Resolution   | 5,000   |
| InterNet                        | 5,000   |
| LLaVA-Rad                       | 5,000   |
| LLaVA                           | 5,000   |
| LLaVA-Med                       | 17,687  |
| MedSAM                          | 6,406   |
| PMC-LLaMA                       | 10,000  |
| RaTE-NER                        | 5,000   |
| SpecialistVLMs                  | 5,000   |
| UltraSAM                        | 5,000   |
| UniGradICON                     | 5,000   |
| Multi-Tool Multi-Round          | 100,000 |
| **Total**                       | **212,449** |

## Overview

1. **Single-Tool Multi-Round**: Basic tool usage (foundation datasets)
2. **Multi-Tool Multi-Round**: Complex workflows across different tools
3. **Multi-Tool Single-Round**: Reasoning inside one round.

## Single-Round Dialogue

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
      "actions": [{"API_name": "UltraSAM", "API_params": {}}],
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
      "value": "The lung segmentation has been completed successfully."
    }
  ]
}
```

## Single-Tool Multi-Round Dialogue

### Structure
```
Daily Dialogue → User Request → Assistant Tool Call → Tool Output → Assistant Final Response → Daily Dialogue
```

### Scenario Types
**Use Case**: Add daily dialogue at the beginning or end of the conversation for diversity.

```json
{
  "conversations": [
    {"from": "human", "value": "Greetings friend"},
    {"from": "gpt", "thoughts": "…", "actions": [], "value": "I'm doing well, thank you."},
    {
      "from": "human",
      "value": "<image>\nSegment the lung regions in this chest X-ray"
    },
    {
      "from": "gpt", 
      "thoughts": "I need to use UltraSAM to segment the lung regions in this chest X-ray image.",
      "actions": [{"API_name": "UltraSAM", "API_params": {}}],
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
      "value": "The lung segmentation has been completed successfully."
    }
    {"from": "human", "value": "That's all -- thanks!"},
  {"from": "gpt", "thoughts": "…", "actions": [], "value": "Glad to help -- take care!"}]}
  ]
}
```

## Multi-Tool Multi-Round Dialogue

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

## Multi-Tool Single-Round Dialogue



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

## Implementation Guidelines

### Image Management Rules

#### Image Field Logic
```python
# Single image
"image": "/path/to/single.jpg"

# Multiple images
# using first image when first input has two images while only one image required after
"image": ["/path/to/image1.jpg", "/path/to/image2.jpg"]
```
