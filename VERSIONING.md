# Automatic File Versioning

The data generator now **automatically versions all output files** to prevent accidental overwrites!

## How It Works

Every generated file includes:
1. **Version number** (v1, v2, v3, etc.)
2. **Category name** (or "all" if generating all categories)

### Filename Format

```
generated_sentences_v{VERSION}_{CATEGORY}.json
```

## Examples

### Single Category Generation

```bash
# First run
python main.py --category "Anxiety" --count 10
# Creates: generated_sentences_v1_Anxiety.json

# Second run
python main.py --category "Anxiety" --count 10
# Creates: generated_sentences_v2_Anxiety.json

# Third run
python main.py --category "Anxiety" --count 15
# Creates: generated_sentences_v3_Anxiety.json
```

### All Categories

```bash
# First run
python main.py --all --count 3
# Creates: generated_sentences_v1_all.json

# Second run
python main.py --all --count 5
# Creates: generated_sentences_v2_all.json
```

### Multiple Specific Categories

```bash
python main.py --categories "Anxiety" "Depression" --count 7
# Creates: generated_sentences_v1_Anxiety_Depression.json
```

### Different Categories = Different Version Sequences

```bash
python main.py --category "Anxiety" --count 10
# Creates: generated_sentences_v1_Anxiety.json

python main.py --category "Depression" --count 10
# Creates: generated_sentences_v1_Depression.json  (v1 because it's a different category!)

python main.py --category "Anxiety" --count 5
# Creates: generated_sentences_v2_Anxiety.json  (v2 continues Anxiety sequence)
```

## Version Tracking

- Versions are tracked **per category/combination**
- Each category has its own version sequence
- "all" has a separate version sequence
- No files will ever be overwritten automatically

## Output Directory Structure

After running multiple generations:

```
data/output/
├── generated_sentences_v1_Anxiety.json
├── generated_sentences_v2_Anxiety.json
├── generated_sentences_v3_Anxiety.json
├── generated_sentences_v1_Depression.json
├── generated_sentences_v1_Psychosis.json
├── generated_sentences_v1_Mania.json
├── generated_sentences_v1_all.json
└── generated_sentences_v2_all.json
```

## Using Batch Scripts

The `generate_all.bat` and `generate_all.sh` scripts now create **separate versioned files** for each category:

```bash
# Windows
generate_all.bat 10

# Creates 4 separate files:
# - generated_sentences_v1_Anxiety.json
# - generated_sentences_v1_Depression.json
# - generated_sentences_v1_Psychosis.json
# - generated_sentences_v1_Mania.json
```

## Custom Output Paths

If you specify a custom output path, versioning is **bypassed**:

```bash
python main.py --category "Anxiety" --count 10 --output my_custom_file.json
# Creates: my_custom_file.json (no versioning)
```

## The --append Flag

The `--append` flag still works but now appends to the **most recent version** of that category:

```bash
python main.py --category "Anxiety" --count 10
# Creates: generated_sentences_v1_Anxiety.json

python main.py --category "Anxiety" --count 5 --append
# Appends to: generated_sentences_v1_Anxiety.json (existing file)
```

However, **we recommend NOT using --append** with versioning. Instead, just run multiple times and let versioning create separate files!

## Benefits

✅ **No accidental overwrites** - Your previous data is always safe
✅ **Clear organization** - Easy to see which file contains which category
✅ **Version history** - Track different generation runs
✅ **Safe experimentation** - Try different settings without losing work

## Finding Your Files

All versioned files are in:
```
data/output/
```

To see all versions:
```bash
# Windows
dir data\output\generated_sentences_v*

# Linux/Mac
ls data/output/generated_sentences_v*
```

To find the latest version for a category:
```bash
# Windows
dir /O-D data\output\generated_sentences_v*_Anxiety.json

# Linux/Mac
ls -t data/output/generated_sentences_v*_Anxiety.json | head -1
```
