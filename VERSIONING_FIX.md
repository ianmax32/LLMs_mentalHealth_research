# Versioning Fix - Automatic File Versioning Now Working

## 🐛 Problem

Previously, when you ran data generation commands multiple times, the output file was being overwritten instead of creating new versioned files.

**Example of the problem:**
```bash
python main.py --category "Anxiety" --count 10
# Created: generated_sentences.json (OVERWRITTEN every time!)

python main.py --category "Anxiety" --count 10
# Overwrote: generated_sentences.json (data lost!)
```

## ✅ Solution

Fixed `main.py` to properly enable automatic versioning. Now each generation creates a **new versioned file** automatically.

### What Changed

**File:** `main.py` (lines 115-151)

**Before:**
```python
output_file = Path(args.output) if args.output else config.OUTPUT_JSON
# This ALWAYS set output_file to something, preventing versioning

generator.generate_and_save(
    output_file=output_file,  # Always had a value
    ...
)
```

**After:**
```python
output_file = Path(args.output) if args.output else None
# Now None if user doesn't specify --output, triggering versioning!

generator.generate_and_save(
    output_file=output_file,  # None triggers auto-versioning
    ...
)
```

## 🎯 How It Works Now

### Automatic Versioning (Default Behavior)

When you run commands **WITHOUT** specifying `--output`:

```bash
# First run
python main.py --category "Anxiety" --count 10
# Creates: data/output/generated_sentences_v1_Anxiety.json

# Second run (same category)
python main.py --category "Anxiety" --count 10
# Creates: data/output/generated_sentences_v2_Anxiety.json

# Third run (same category)
python main.py --category "Anxiety" --count 10
# Creates: data/output/generated_sentences_v3_Anxiety.json

# Different category - separate version sequence!
python main.py --category "Depression" --count 10
# Creates: data/output/generated_sentences_v1_Depression.json
```

### Manual File Naming (Optional)

If you **DO** specify `--output`, it uses your exact filename (no versioning):

```bash
python main.py --category "Anxiety" --count 10 --output data/my_custom_file.json
# Creates: data/my_custom_file.json (overwrites if exists)
```

## 📋 Versioning Rules

1. **Each category has its own version sequence**
   - `generated_sentences_v1_Anxiety.json`
   - `generated_sentences_v1_Depression.json`
   - `generated_sentences_v1_Psychosis.json`
   - `generated_sentences_v1_Mania.json`

2. **Version numbers auto-increment**
   - Scans for existing files matching the pattern
   - Finds the highest version number
   - Creates next version (max + 1)

3. **All categories together**
   ```bash
   python main.py --all --count 10
   # Creates: generated_sentences_v1_all.json

   python main.py --all --count 10
   # Creates: generated_sentences_v2_all.json
   ```

4. **Multiple specific categories**
   ```bash
   python main.py --categories Anxiety Depression --count 10
   # Creates: generated_sentences_v1_Anxiety_Depression.json
   ```

## 🚀 Usage Examples

### Example 1: Generate for one category multiple times

```bash
# Each run creates a new versioned file
python main.py --category "Anxiety" --count 10
python main.py --category "Anxiety" --count 10
python main.py --category "Anxiety" --count 10

# Results:
# data/output/generated_sentences_v1_Anxiety.json
# data/output/generated_sentences_v2_Anxiety.json
# data/output/generated_sentences_v3_Anxiety.json
```

### Example 2: Use batch script

```bash
# Each category gets its own versioned file
generate_all.bat 10

# Results:
# data/output/generated_sentences_v1_Anxiety.json
# data/output/generated_sentences_v1_Depression.json
# data/output/generated_sentences_v1_Psychosis.json
# data/output/generated_sentences_v1_Mania.json

# Run again - versions increment!
generate_all.bat 10

# Results:
# data/output/generated_sentences_v2_Anxiety.json
# data/output/generated_sentences_v2_Depression.json
# data/output/generated_sentences_v2_Psychosis.json
# data/output/generated_sentences_v2_Mania.json
```

### Example 3: Generate all categories together

```bash
python main.py --all --count 10

# Creates: generated_sentences_v1_all.json

python main.py --all --count 10

# Creates: generated_sentences_v2_all.json
```

## 🔍 Verification

To verify the fix is working, check your `data/output/` folder after each generation:

```bash
# Windows
dir data\output\

# You should see files like:
# generated_sentences_v1_Anxiety.json
# generated_sentences_v2_Anxiety.json
# generated_sentences_v1_Depression.json
# etc.
```

## ⚠️ Important Notes

1. **Automatic versioning only works when you DON'T specify --output**
   - ✅ `python main.py --category "Anxiety" --count 10` → Auto-versioned
   - ❌ `python main.py --category "Anxiety" --count 10 --output my_file.json` → No versioning

2. **Append mode still works with --append flag**
   ```bash
   python main.py --category "Anxiety" --count 10 --output existing.json --append
   # Appends to existing.json without versioning
   ```

3. **Version numbers are based on EXISTING files**
   - If you delete files, version numbers may reuse previous numbers
   - This is intentional to fill gaps in version sequences

## 🎉 Benefits

✅ **No more overwriting** - Each generation gets a new file
✅ **Automatic organization** - Files are clearly labeled with version and category
✅ **Data safety** - Previous generations are preserved
✅ **Easy tracking** - Know which generation each file came from

## 📁 File Structure Example

After multiple generations:

```
data/output/
├── generated_sentences_v1_Anxiety.json       # 10 sentences
├── generated_sentences_v2_Anxiety.json       # 10 more sentences
├── generated_sentences_v3_Anxiety.json       # 10 more sentences
├── generated_sentences_v1_Depression.json    # 10 sentences
├── generated_sentences_v2_Depression.json    # 10 more sentences
├── generated_sentences_v1_Psychosis.json     # 10 sentences
├── generated_sentences_v1_Mania.json         # 10 sentences
└── generated_sentences_v1_all.json           # All categories together
```

## 🧪 Testing

You can test the versioning system:

```bash
# Run the test script
python test_versioning_fix.py

# Or test manually
python main.py --category "Anxiety" --count 3
python main.py --category "Anxiety" --count 3
python main.py --category "Anxiety" --count 3

# Check data/output/ - you should see v1, v2, v3
```

---

**The versioning issue is now fixed! Each generation creates a new file automatically.** 🎉
