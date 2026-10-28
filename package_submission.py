"""Package only coursework outputs; never include credentials, local environments or data."""
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parent
def main():
    files=[]
    for pattern in ['*.py','requirements*.txt','pytest.ini','README.md','.gitignore','status.json','results.csv','per_class.csv','tests/*.py','runs/*/*.json','runs/*/*.csv','runs/*/*.npz','report/*.tex','report/*.txt','report/*.pdf','report/figures/*.pdf']:
        files.extend(ROOT.glob(pattern))
    files=sorted(set(p for p in files if p.is_file() and p.name!='finalize_when_done.py'))
    with zipfile.ZipFile(ROOT/'GitHub_Submission.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(ROOT))
    manifest=[str(p.relative_to(ROOT)).replace('\\','/') for p in files]
    (ROOT/'submission_files.json').write_text(json.dumps(manifest,indent=2))
    print(f'Packaged {len(files)} files')
if __name__=='__main__': main()
