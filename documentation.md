The pipeline now does PDF → parse text/tables/figures → LLM extraction → CSV/parquet in one command. You no longer need the separate scientific-pdf-pipeline.

What happens (simple picture)
You put PDFs in a folder
The code reads the PDFs (text + tables + images)
Local AI models (Ollama) pull out the structured data
You get a spreadsheet-like file you can open
One-time setup (PowerShell)
Open PowerShell and run these one by one.

Step 1 — Go to the project folder
cd "C:\Users\HP\Downloads\new\3. LLM"
Step 2 — Install Python packages
python -m pip install -e ".[dev]"
Step 3 — Install Ollama (the local AI app)
Download and install from: https://ollama.com/download
Open a new PowerShell window and start it (if it is not already running):
ollama serve
Leave that window open. Open a second PowerShell for the next steps.

Step 4 — Download the AI models
In the second PowerShell:

ollama pull granite4.1:8b
ollama pull gemma3:12b
ollama pull qwen3:8b
These match config.yaml. They are large and can take a while.

Smaller PC tip: edit config.yaml and set primary, cross_check, and tie_breaker all to the same small model, e.g. llama3.2:3b, then only run:

ollama pull llama3.2:3b
Step 5 — Put your PDFs in the input folder
cd "C:\Users\HP\Downloads\new\3. LLM"
explorer .\input_pdfs
Copy your .pdf files into that folder (text PDFs work best; scanned image-only PDFs will not extract well).

Run the full extraction
Still in the project folder:

cd "C:\Users\HP\Downloads\new\3. LLM"
llm-extract from-pdfs --config config.yaml
Or use the helper script:

cd "C:\Users\HP\Downloads\new\3. LLM"
.\run_pipeline.ps1
If Windows blocks the script:

Set-ExecutionPolicy -Scope Process Bypass
.\run_pipeline.ps1
Where your results are
File	What it is
extraction_output\extracted_dataset.csv
Main table (open in Excel)
extraction_output\extracted_dataset.parquet
Same data for Python/ML
pipeline_output\
Parsed text/tables per PDF
input_pdfs\
Your original PDFs
Open the CSV:

explorer .\extraction_output\extracted_dataset.csv
Useful extras
Change topic (SAF/HEFA, biodiesel, etc.) — edit one line in config.yaml:

extraction:
  schema_name: "saf_hefa"   # or biodiesel / methanol_synthesis / mof_defect_engineering
Re-run after adding more PDFs — same command; already-done papers are skipped.

Turn on chart/figure reading later — in config.yaml set enable_figure_vision: true (needs a vision model like gemma3:12b).

Checklist if something fails
Are you in 3. LLM when you run the command?
Is ollama serve running?
Does ollama list show the model names from config.yaml?
Are there .pdf files in input_pdfs?
That is the whole beginner path: install once → drop PDFs → run one command → open the CSV.