import os
import pandas as pd

OUTDIR = r"C:\Users\giaco\Desktop\Git_META\META\results\RQ1_3_GM"
OUTPUT_XLSX = os.path.join(OUTDIR, "all_overlap_results.xlsx")

# Mappa per abbreviazioni fogli
ABBREV_MAP = {
    "ADULTS_ALLDISORDERS_CTX": "AD_CTX",
    "ADULTS_ALLDISORDERS_SCTX": "AD_SCTX",
    "ADOLESCENTS_ALLDISORDERS_CTX": "ADO_CTX",
    "ADOLESCENTS_ALLDISORDERS_SCTX": "ADO_SCTX",
    "ADULTS_CLUSTER_Psychotic_CTX": "CL_PSY_CTX",
    "ADULTS_CLUSTER_Psychotic_SCTX": "CL_PSY_SCTX",
    "ADULTS_CLUSTER_Neurodevelopmental_CTX": "CL_ND_CTX",
    "ADULTS_CLUSTER_Neurodevelopmental_SCTX": "CL_ND_SCTX",
    "ADULTS_CLUSTER_AN_OCD_CTX": "CL_ANOCD_CTX",
    "ADULTS_CLUSTER_AN_OCD_SCTX": "CL_ANOCD_SCTX",
    "ADULTS_CLUSTER_Mood_Anxiety_CTX": "CL_MA_CTX",
    "ADULTS_CLUSTER_Mood_Anxiety_SCTX": "CL_MA_SCTX",
}

csv_files = [f for f in os.listdir(OUTDIR) if f.endswith(".csv")]

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(OUTDIR, csv_file))
        base_name = os.path.splitext(csv_file)[0]

        # Trova la chiave corrispondente nella mappa (prima parte del nome)
        for key in ABBREV_MAP:
            if base_name.startswith(key):
                abbrev = ABBREV_MAP[key]
                # aggiungi top10/top20 dal nome
                if "top10" in base_name:
                    sheet_name = f"{abbrev}_top10"
                elif "top20" in base_name:
                    sheet_name = f"{abbrev}_top20"
                else:
                    sheet_name = abbrev
                break
        else:
            # fallback: usa solo i primi 31 caratteri
            sheet_name = base_name[:31]

        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"✅ Tutti i CSV uniti in {OUTPUT_XLSX}")

