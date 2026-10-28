from pathlib import Path
import urllib.request
import zipfile

URL='https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip'
def main():
    p=Path(__file__).resolve().parent/'data/raw/opportunity.zip'
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        tmp=p.with_suffix('.part')
        urllib.request.urlretrieve(URL,tmp)
        with zipfile.ZipFile(tmp) as z:
            if z.testzip(): raise RuntimeError('Corrupt archive')
        tmp.replace(p)
    print(p)
if __name__=='__main__': main()
