# ------------------------------
# zusammenfassung.py
# Zusammenfassende Auswertung
# ------------------------------

from collections import defaultdict
import xml.etree.ElementTree as ET

from hilfsfunktoinen import (
    sortiere_dana,
    lade_referenz_liste_aus_xsd
)

# ------------------------------
# Hilfsfunktionen
# ------------------------------

def label_url(lbl):
    return f'{{www.dglab.uni-jena.de/vmf/{lbl}}}ana'


# ------------------------------
# Text-IDs bestimmen
# ------------------------------

def finde_text_ids(root, typ_name=None, NS=None):
    text_ids = []

    for corp in root.findall('.//tei:teiCorpus', NS):
        for tei in corp.findall('.//tei:TEI', NS):
            for text in tei.findall('.//tei:text', NS):
                text_id = text.attrib.get(
                    '{http://www.w3.org/XML/1998/namespace}id', ''
                )

                if typ_name is None:
                    text_ids.append(text_id)
                    continue

                gefunden = False
                for seg in text.findall('.//tei:seg', NS):
                    tn = seg.attrib.get(label_url('a'), '')
                    for x in tn.replace(';', ',').split(','):
                        if x.strip() == typ_name:
                            gefunden = True
                            break
                    if gefunden:
                        break

                if gefunden:
                    text_ids.append(text_id)

    return sorted(set(text_ids))


# ------------------------------
# Zeilen extrahieren
# ------------------------------

def extrahiere_zeilen(root, text_ids, referenz_liste, NS, typ_name=None):
    daten = defaultdict(list)

    for text in root.findall('.//tei:text', NS):
        text_id = text.attrib.get(
            '{http://www.w3.org/XML/1998/namespace}id'
        )
        if text_id not in text_ids:
            continue

        for seg in text.findall('.//tei:seg', NS):
            labela = seg.attrib.get(label_url('a'), 'N')
            if not labela.startswith('a'):
                continue

            labelbs = [seg.attrib.get(label_url(f'b{i}'), 'N') for i in range(1, 6)]
            labelcs = [seg.attrib.get(label_url(f'c{i}'), 'N') for i in range(1, 6)]
            labelds = [seg.attrib.get(label_url(f'd{i}'), 'N') for i in range(1, 6)]

            dana = '_'.join(labelds).replace('_N', '')
            try:
                labeld = sortiere_dana(dana, referenz_liste)
            except ValueError:
                labeld = dana

            for lb, lc in zip(labelbs, labelcs):
                row = f"{labela}:{lb}:{lc}:{labeld}"
                if ':N:' not in row:
                    if typ_name is not None and labela == typ_name:
                        row = f"→{row}"
                    daten[text_id].append(row)

    return daten


# ------------------------------
# Hauptauswertung
# ------------------------------

def zusammenfassung_notebook(Repertoire=None, Gesuchter_Typ=None):
    NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

    if Repertoire is None:
        Repertoire = 'test.xml'

    root = ET.parse(Repertoire).getroot()
    referenz_liste = lade_referenz_liste_aus_xsd('kf/vmf_d1.xsd')

    # --- vmf_a.xsd laden ---
    replacement_dict = {}
    tree = ET.parse('kf/vmf_a.xsd')
    root_xsd = tree.getroot()
    ns_xsd = {'xs': 'http://www.w3.org/2001/XMLSchema'}

    for enum in root_xsd.findall(".//xs:enumeration", ns_xsd):
        value = enum.attrib.get('value', '')
        if ' ' in value:
            key = value.split(' ', 1)[0]
            replacement_dict[key] = value

    text_ids = finde_text_ids(root, Gesuchter_Typ, NS)
    daten = extrahiere_zeilen(root, text_ids, referenz_liste, NS, Gesuchter_Typ)

    output = []
    anzahl_mit_treffern = 0
    anzahl_ohne_treffern = 0

    # --- Kopf ---
    if Gesuchter_Typ is None:
        output.append("Auswertung aller gelabelten Zeilen\n")
    else:
        output.append(f"Der gesuchte Typ = {Gesuchter_Typ}\n")

    output.append("=============================\n")

    # --- Pro Text ---
    for tid in text_ids:
        output.append(f"\nText_ID: {tid}\n")

        labela_set = set()
        for text in root.findall('.//tei:text', NS):
            if text.attrib.get(
                '{http://www.w3.org/XML/1998/namespace}id'
            ) != tid:
                continue

            for seg in text.findall('.//tei:seg', NS):
                la = seg.attrib.get(label_url('a'), 'N')
                if la.startswith('a'):
                    labela_set.add(la)

        for la in sorted(labela_set):
            output.append(f"  {replacement_dict.get(la, la)}\n")

        if labela_set:
            output.append('-' * (6 + len(tid)) + "\n")

        treffer = daten.get(tid)

        if treffer:
            anzahl_mit_treffern += 1
            for z in treffer:
                output.append(z + "\n")
        else:
            anzahl_ohne_treffern += 1
            output.append("(keine Treffer)\n")

    return (
        ''.join(output),
        anzahl_mit_treffern,
        anzahl_ohne_treffern
    )


# ------------------------------
# Benutzerabfrage
# ------------------------------

def frage_typ_ab():
    print("Auswertung starten")
    print("------------------")
    print("↵ Enter → alle Textannotationen")
    print("Gesuchter Typ eingeben, z.B. a300 → Textannotationen nur für den angegebenen Typ\n")

    eingabe = input("Gesuchter Typ: ").strip()
    return eingabe or None


# ------------------------------
# Zusammenfassung anzeigen
# ------------------------------

def zeige_zusammenfassung(mit_treffern, ohne_treffern):
    print("\n=============================")
    print("Zusammenfassung:")
    print(f"Gelabelte Texte: {mit_treffern}")
    print(f"Nicht gelabelte Texte: {ohne_treffern}")
    print(f"Gesamtzahl Texte: {mit_treffern + ohne_treffern}")

