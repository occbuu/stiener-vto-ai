"""
Build the curated, pseudonymised Steiner-stick dataset from private sources.

Sources (folder data-steiner/):
  * DATA STEINER STICK NCS_v0.1.xlsx  - manual transcription (500 rows)
  * Steiner Stick TS *.pdf            - original VTO worksheets (501 pages, Canva export)

Output (ai/data/processed/):
  * steiner_vto_clean.csv   - one row per worksheet, coded id, no names
  * data_audit_log.csv      - every correction / flag with its provenance

Requires poppler `pdftotext` on PATH (Windows: conda install -c conda-forge poppler).
Patient names printed on the PDFs are not read into the output. Treat all
patient-level outputs as restricted research data even when names are absent.

Usage:  python scripts/build_dataset.py --raw ../data-steiner --out data/processed
"""
import argparse, glob, os, re, subprocess
import numpy as np, pandas as pd

CEPH = ['SNA', 'SNB', 'ANB', 'GoGn_SN', 'U1NA_mm', 'U1NA_deg', 'L1NB_mm', 'L1NB_deg', 'U1L1', 'PogNB']
STICK = ['CL', 'P_L1', 'SPEE', 'P_L6', 'EXP', 'IPR', 'EXT']
XLSX_COLS = ['stt', 'code'] + STICK + ['TOTAL'] + CEPH + ['ANBp', 'PogNBp', 'L1NBp']
LABELS = {'SNA': r'Góc SNA', 'SNB': r'Góc SNB', 'ANB': r'\n\s*ANB', 'GoGn_SN': r'Góc SN-GoGn',
          'U1NA_mm': r'U1 - NA \(mm \)', 'U1NA_deg': r'U1 - NA \(Góc\)', 'L1NB_mm': r'L1 - NB \(mm\)',
          'L1NB_deg': r'L1 - NB \(Góc\)', 'U1L1': r'Góc \(U1-L1\)', 'PogNB': r'\n\s*Pog-NB'}
# y-bands (pt) of the space-analysis table on the A4 worksheet; x<200 = "+" column, 200-233 = "-" column
BANDS = [('CL', 495, 538), ('P_L1', 540, 578), ('SPEE', 580, 622), ('P_L6', 624, 682),
         ('EXP', 684, 718), ('IPR', 720, 757), ('EXT', 758, 802)]
NUM = re.compile(r'-?\d+(\.\d+)?')


def _pdftotext(args):
    return subprocess.run(['pdftotext'] + args, capture_output=True, text=True, encoding='utf-8').stdout


def parse_pdfs(raw_dir):
    rows = []
    for f in glob.glob(os.path.join(raw_dir, 'Steiner Stick TS*.pdf')):
        first = int(re.search(r'TS (\d+)', f).group(1))
        lay = _pdftotext(['-layout', f, '-']).split('\f')
        bb = _pdftotext(['-bbox', f, '-']).split('<page ')[1:]
        for pos, (p, pg) in enumerate(zip(lay, bb)):
            lab = re.search(r'Mã số\s*(TS\s*\d+)', p)
            r = {'pdf_file': os.path.basename(f), 'pdf_page': pos + 1, 'seq': first + pos,
                 'pdf_label': lab.group(1).replace(' ', '') if lab else None}
            s = re.search(r'Giới\s*:?\s*(Nam|NAM|nam|Nữ|NỮ|nữ)', p)
            a = re.search(r'Tuổi\s*:?\s*(\d{1,2})\b', p)
            r['sex'] = ('F' if s.group(1).lower() == 'nữ' else 'M') if s else None
            r['age'] = int(a.group(1)) if a else np.nan
            for k, l in LABELS.items():
                m = re.search(l + r'\s+(-?\d+(?:[.,]\d+)?)\b', p)
                r[k] = float(m.group(1).replace(',', '.')) if m else np.nan
            words = [(float(a_), float(b_), w) for a_, b_, w in re.findall(
                r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="[\d.]+" yMax="[\d.]+">([^<]*)</word>', pg)]
            for name, y0, y1 in BANDS:
                plus = [float(w) for x, y, w in words if y0 <= y < y1 and 150 <= x < 200 and NUM.fullmatch(w)]
                minus = [float(w) for x, y, w in words if y0 <= y < y1 and 200 <= x < 233 and NUM.fullmatch(w)]
                r[name] = sum(plus) - sum(minus)
            rows.append(r)
    return pd.DataFrame(rows).sort_values(['seq', 'pdf_page']).reset_index(drop=True)


def main(raw_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw = pd.read_excel(os.path.join(raw_dir, 'DATA STEINER STICK NCS_v0.1.xlsx'), header=None, skiprows=3).iloc[:, :23]
    raw.columns = XLSX_COLS
    x = raw.apply(pd.to_numeric, errors='coerce')
    pdf = parse_pdfs(raw_dir)
    # The 201-250 file has 51 pages (label TS206 printed twice); the Excel follows page order and
    # stops before the last page of that file. Map Excel row k -> page with seq k in its own file.
    pdf['file_first'] = pdf.pdf_file.str.extract(r'TS (\d+)').astype(int)
    log, recs = [], []

    def note(i, field, old, new, src, reason):
        log.append(dict(record=i, field=field, original=str(old), corrected=new, source=src, reason=reason))

    for i, r in x.iterrows():
        k = int(r.code)
        blk = (k - 1) // 50 * 50 + 1
        cand = pdf[(pdf.file_first == blk) & (pdf.pdf_page == k - blk + 1)]
        p = cand.iloc[0] if len(cand) else None
        rec = {c: r[c] for c in STICK + CEPH + ['ANBp', 'PogNBp', 'L1NBp']}
        rec.update(src_xlsx_row=k, pdf_file=p.pdf_file if p is not None else None,
                   pdf_page=p.pdf_page if p is not None else None, pdf_label=p.pdf_label if p is not None else None,
                   sex=p.sex if p is not None else None, age=p.age if p is not None else np.nan, flags='')
        rid = 'SS%03d' % (len(recs) + 1)
        for f in CEPH:
            v, rv = r[f], raw.at[i, f]
            isdate = hasattr(rv, 'year')
            if (pd.isna(v) or isdate) and p is not None and not pd.isna(p[f]):
                rec[f] = p[f]; note(rid, f, rv, p[f], f'{p.pdf_file} p{p.pdf_page}', 'Excel date auto-format / empty cell')
        # physiologically impossible values -> derive from the SNA-SNB-ANB identity
        if not (60 < rec['SNA'] < 110):
            new = round(rec['SNB'] + rec['ANB'], 2); note(rid, 'SNA', rec['SNA'], new, 'SNB+ANB', 'typo in source (value <60°)'); rec['SNA'] = new
        if not (60 < rec['SNB'] < 110):
            new = round(rec['SNA'] - rec['ANB'], 2); note(rid, 'SNB', rec['SNB'], new, 'SNA-ANB', 'typo in source (value <60°)'); rec['SNB'] = new
        if rec['ANB'] == 0 and abs(rec['SNA'] - rec['SNB']) > 0.5:
            new = round(rec['SNA'] - rec['SNB'], 2); note(rid, 'ANB', 0, new, 'SNA-SNB', 'ANB recorded as 0 although SNA-SNB differs (negative ANB truncated)'); rec['ANB'] = new
        for f in STICK:
            if hasattr(raw.at[i, f], 'year') or pd.isna(rec[f]):
                rec[f] = np.nan
        s_x = sum(0 if pd.isna(rec[f]) else rec[f] for f in STICK)
        pdf_ok = p is not None and abs(sum(p[f] for f in STICK)) < 0.01
        differs = p is not None and any(pd.isna(rec[f]) or abs(rec[f] - p[f]) > 0.01 for f in STICK)
        ext_conflict = p is not None and (rec['EXT'] > 0) != (p['EXT'] > 0)
        if (abs(s_x) > 0.01 or any(pd.isna(rec[f]) for f in STICK) or ext_conflict) and pdf_ok and differs:
            for f in STICK:
                if pd.isna(rec[f]) or abs(rec[f] - p[f]) > 0.01:
                    note(rid, f, rec[f], p[f], f'{p.pdf_file} p{p.pdf_page}', 'space table unbalanced / column-shifted in Excel; PDF balanced')
                    rec[f] = p[f]
        for f in STICK:
            if pd.isna(rec[f]): rec[f] = 0.0; note(rid, f, 'empty', 0.0, 'rule', 'empty cell in balanced table = 0')
        if abs(sum(rec[f] for f in STICK)) > 0.01:
            rec['flags'] += 'space_unbalanced;'; note(rid, '__flag__', sum(rec[f] for f in STICK), None, '-', 'space table does not sum to 0 in either source')
        rec['id'] = rid
        recs.append(rec)
    df = pd.DataFrame(recs)
    # duplicated cephalometric block (Canva template not updated) -> flag the later record
    key = df[CEPH].round(2).astype(str).agg('|'.join, axis=1)
    for rid in df.loc[key.duplicated(keep='first'), 'id']:
        df.loc[df.id == rid, 'flags'] += 'ceph_duplicate_of_previous_sheet;'
        note(rid, '__flag__', None, None, '-', 'cephalometric block identical to an earlier sheet; ceph values unreliable')
    # page present in PDF but not transcribed in Excel
    used = set(zip(df.pdf_file, df.pdf_page))
    for _, p in pdf.iterrows():
        if (p.pdf_file, p.pdf_page) not in used:
            rid = 'SS%03d' % (len(df) + 1)
            rec = {c: p[c] for c in STICK + CEPH}
            rec.update(ANBp=np.nan, PogNBp=np.nan, L1NBp=np.nan, src_xlsx_row=np.nan, pdf_file=p.pdf_file,
                       pdf_page=p.pdf_page, pdf_label=p.pdf_label, sex=p.sex, age=p.age, id=rid,
                       flags='pdf_only;' + ('' if abs(sum(p[f] for f in STICK)) < .01 else 'space_unbalanced;'))
            df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
            note(rid, '__added__', None, None, f'{p.pdf_file} p{p.pdf_page}', 'worksheet missing from Excel, parsed from PDF')
    df['extraction'] = (df.EXT > 0).astype(int)
    df['usable_ceph'] = ~df['flags'].str.contains('ceph_duplicate')
    df['usable_plan'] = ~df['flags'].str.contains('space_unbalanced')
    cols = ['id', 'src_xlsx_row', 'pdf_file', 'pdf_page', 'pdf_label', 'sex', 'age'] + CEPH + ['CL', 'SPEE'] + \
           ['P_L1', 'P_L6', 'EXP', 'IPR', 'EXT', 'extraction', 'ANBp', 'PogNBp', 'L1NBp', 'usable_ceph', 'usable_plan', 'flags']
    df[cols].to_csv(os.path.join(out_dir, 'steiner_vto_clean.csv'), index=False)
    pd.DataFrame(log).to_csv(os.path.join(out_dir, 'data_audit_log.csv'), index=False)
    print(f'{len(df)} records written; {len(log)} audit entries')
    print(df['flags'].value_counts())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='../data-steiner'); ap.add_argument('--out', default='data/processed')
    a = ap.parse_args(); main(a.raw, a.out)
