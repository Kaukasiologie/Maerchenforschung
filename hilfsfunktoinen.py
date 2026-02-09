from typing import Tuple
from numpy.typing import ArrayLike
from typing import Counter
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

################################################################################
# ############################ Dateien parsen ##################################
################################################################################

def lade_referenz_liste_aus_xsd(dateipfad, typname="ana", ns="www.dglab.uni-jena.de/vmf/d1"):
    """Lädt die Werte aus xs:enumeration für den angegebenen SimpleType aus der XSD."""
    tree = ET.parse(dateipfad)
    root = tree.getroot()
    # Namespace-Deklaration
    nsmap = {
        "xs": "http://www.w3.org/2001/XMLSchema",
        "d1": ns
    }

    # Gesuchten simpleType finden
    simple_type = root.find(
        f".//xs:simpleType[@name='{typname}']", namespaces=nsmap)
    if simple_type is None:
        raise ValueError(f"Kein simpleType '{typname}' gefunden.")

    # Alle Enumeration-Werte sammeln
    enumeration_werte = [
        e.attrib["value"] for e in simple_type.findall(".//xs:enumeration", namespaces=nsmap)
    ]
    return enumeration_werte

def sortiere_dana(dana_str, referenz_liste):
    """Sortiert gültige Einträge aus dana_str gemäß der referenz_liste.
    Unbekannte Einträge werden entfernt, doppelte behalten."""

    eintraege = dana_str.split('_')
    gueltige = [e for e in eintraege if e in referenz_liste]
    sortiert = sorted(gueltige, key=lambda x: referenz_liste.index(x))

    return '_'.join(sortiert)

def sonderzeichen_entfernen(text: str):
    res = (text.lower().strip()
           .replace('|', '').replace(':', '').replace("ä", "ae")
           .replace("ü", "ue").replace("ö", "oe").replace("ß", "ss")
           .replace(",", "").replace("\n", " ").replace("'", "")
           .replace("…", "").replace('"', "").replace("*", "")
           .replace("(", "").replace(")", "").replace("-", "")
           .replace("]", "").replace("[", "").replace(".", "")
           .replace("?", "").replace("!", "").replace("„", "")
           .replace("+", "").replace("=", "").replace("_", "")
           .replace("­", "").replace('“', "").replace('”', "")
           .replace(";", "").replace('~', "").replace('`', "")
           .replace("«", "").replace("»", ""))

    while res.endswith(' '):
        res = res[:-1]
    return res

def parse_xml_kern(tei, namespace, referenz_liste, parse_labels):
    LABELS = ['a', 'b1', 'b2', 'b3', 'b4', 'b5', 'c1', 'c2', 'c3', 'c4', 'c5', 'd1', 'd2', 'd3', 'd4', 'd5']
    LABEL_PAAR_KEY_LISTE = [['b1', 'c1'], ['b2', 'c2'], ['b3', 'c3'], ['b4', 'c4'], ['b5', 'c5']]
    maerchen = ''
    for ganze in tei.findall(".//tei:text", namespace):
        quelle = ganze.attrib.get('{http://www.w3.org/XML/1998/namespace}id', '')
        for body in ganze.findall(".//tei:body", namespace):
            for absatz in body.findall(".//tei:p", namespace):
                for phrase in absatz.findall(".//tei:seg", namespace):

                    if parse_labels:
                        label_dict = dict()
                        for label in LABELS:
                            label_wert = phrase.attrib.get(f'{{www.dglab.uni-jena.de/vmf/{label}}}ana', 'N/A')
                            label_dict[label] = label_wert

                        if not label_dict['a'].startswith('a'):
                            continue
                        if label_dict['b1'] == 'N':
                            continue

                        dana = f"{label_dict['d1']}_{label_dict['d2']}_{label_dict['d3']}_{label_dict['d4']}_{label_dict['d5']}"

                        try:
                            sortierte_dana = sortiere_dana(dana, referenz_liste).replace('_N', '')
                        except ValueError as e:
                            sortierte_dana = dana
                            print(f"Warnung bei {quelle}: {e}")

                        if sortierte_dana == 'N':
                            continue

                        label_gefilterte_liste = []

                        for lb_key, lc_key in LABEL_PAAR_KEY_LISTE:
                            lb = label_dict[lb_key]
                            lc = label_dict[lc_key]

                            if lb == 'N' or lc == 'N':
                                continue

                            label_gefilterte_liste.append(label_dict['a'])
                            label_gefilterte_liste.append(lb)
                            label_gefilterte_liste.append(lc)
                            label_gefilterte_liste.append(sortierte_dana)

                        if len(label_gefilterte_liste) == 0:
                            continue

                        label_gesammt = f"{':'.join(label_gefilterte_liste)}:"

                        inhalt = sonderzeichen_entfernen(phrase.text)
                        maerchen += f'{quelle},{label_gesammt},{inhalt},0\n'

                    else:
                        labelb1 = phrase.attrib.get('{www.dglab.uni-jena.de/vmf/b1}ana', 'N/A')
                        if labelb1.startswith('N'):
                            inhalt = sonderzeichen_entfernen(phrase.text)
                            maerchen += f'{quelle},{inhalt},0\n'
    return maerchen


def parse_xml(root_node, namespace, referenz_liste, parse_labels):
    maerchen = ""
    for corp in root_node.findall(".//tei:teiCorpus", namespace):
        for tei in corp.findall("tei:TEI", namespace):
            maerchen += parse_xml_kern(tei, namespace, referenz_liste, parse_labels)

    return maerchen

################################################################################
# ########################## Daten preparieren #################################
################################################################################

def gesuchte_episoden_labeln(gesuchte_episode, df: pd.DataFrame):
    for idx, index_string in zip(df.index, df.index_string):
        if gesuchte_episode in index_string:
            df.loc[idx, 'index_binar'] = 1

################################################################################
# ########################## Machine learning ##################################
################################################################################

def find_similar(model_vec, vocabulary_vec)-> Tuple[np.ndarray, np.ndarray]:
    similarities = cosine_similarity(model_vec, vocabulary_vec).flatten()
    similarities = np.delete(similarities, -1)
    sorted_index = similarities.argsort()[::-1]
    similarities = similarities[sorted_index]

    return sorted_index, similarities

def get_scored_indices(similarities, factor):
    max_score = max(score for _, score in similarities)

    indices = []
    scores = []
    for index, score in similarities:
        if score > factor * max_score:
            indices.append(index)
            scores.append(score)
    return indices, scores

def create_logistic_regression_model(data, label, min_df, custom_stop_word_list, ngram_range,
                                     perform_regularization_sweep=False):
    """
    Wir vektorisieren die Daten mit TF-IDF vektoriser und kategorisieren sie dann mit logistisher regression.
    Wir machen eine Kreuzvalidierung zwischen mehreren Logistichen regressionsmodellen mit verschiedenen
    regularisierungsparameter.
    Nach diedem prozess wählen wir eins mit dem bessten "Score."

    """

    pipe = make_pipeline(
        TfidfVectorizer(min_df=min_df, stop_words=custom_stop_word_list, ngram_range=ngram_range, norm=None),
        LogisticRegression()
    )

    if perform_regularization_sweep:

        param_grid = {'logisticregression__C': [0.01, 0.1, 1]}
        grid = GridSearchCV(pipe, param_grid, cv=5)
        grid.fit(data, label)
        return grid.best_estimator_
    else:
        pipe.fit(data, label)
        return pipe

def feature_set_selection(coefficients, features):

    # Sortieren der Indizes der Koeffizienten nach ihrer Größe
    sorted_coef_index = np.argsort(coefficients)

    # Extrahieren der Top-Merkmale mit den höchsten Koeffizienten
    sorted_features = features[sorted_coef_index]
    sorted_coefficients = coefficients[sorted_coef_index]

    pos_mask = sorted_coefficients > sorted_coefficients[-1]*(1e-3)
    pos_coeffs = sorted_coefficients[pos_mask]
    pos_features = sorted_features[pos_mask]

    return pos_coeffs, pos_features
