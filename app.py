"""
Volume Alto — recomendador musical (Flask)
Bibliotecas além do Flask: pandas, numpy, scikit-learn, rapidfuzz.

- pandas: lê o catálogo (data/artistas.csv)
- scikit-learn: TF-IDF das tags + similaridade de cosseno
- numpy: perfil de gosto (média dos artistas escolhidos) e ranking
- rapidfuzz: entende nomes digitados com erro ("limp bizquits")
As fotos dos artistas são buscadas pelo navegador na Wikipédia (ver index.html).
"""
import unicodedata
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
BASE = Path(__file__).parent


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in texto if not unicodedata.combining(c)).lower().strip()


# ---------- catálogo e modelo ----------
df = pd.read_csv(BASE / "data" / "artistas.csv", keep_default_na=False)
df["chave"] = df["nome"].map(normalizar)
df["origem"] = np.where(df["pais"] == "BR", "brasil", "internacional")
df["texto"] = (df["tags"] + " " + df["genero"].str.replace("/", " ") + " " + df["origem"])
EH_UNDERGROUND = df["tags"].str.contains("underground|indie|lo-fi").to_numpy()
EH_BR = (df["pais"] == "BR").to_numpy()

vetorizador = TfidfVectorizer(token_pattern=r"[\wÀ-ÿ]+", sublinear_tf=True)
MATRIZ = vetorizador.fit_transform(df["texto"])
VOCAB = np.array(vetorizador.get_feature_names_out())
NOMES = df["nome"].tolist()
CHAVES = df["chave"].tolist()

HUMORES = {
    "": "Qualquer clima",
    "pesado": "Pesado",
    "melancólico": "Melancólico",
    "psicodelia": "Psicodélico",
    "irônico": "Irônico",
    "dançante": "Dançante",
    "intimista": "Intimista",
    "nostálgico": "Nostalgia anos 2000",
    "clássico": "Clássicos",
}
ESCOPOS = {"": "Brasil e exterior", "br": "Só Brasil", "int": "Só exterior"}
SUGESTOES = ["Ana Frango Elétrico", "Sophia Chablau e Uma Enorme Perda de Tempo", "Besouro Mulher",
             "Zé Ibarra", "Vitor Só", "Limp Bizkit", "Korn", "System of a Down", "Linkin Park",
             "Los Hermanos", "Pitty", "Deftones"]


def achar_artista(entrada: str, corte: int = 72):
    r = process.extractOne(normalizar(entrada), CHAVES, scorer=fuzz.WRatio, score_cutoff=corte)
    return r[2] if r else None


def recomendar(entradas, humor="", escopo="", underground=False, limite=9):
    indices, nao_achados = [], []
    for e in entradas:
        i = achar_artista(e)
        if i is None:
            nao_achados.append(e)
        elif i not in indices:
            indices.append(i)
    if not indices:
        return [], [], nao_achados

    perfil = np.asarray(MATRIZ[indices].mean(axis=0))
    notas = cosine_similarity(perfil, MATRIZ).ravel()

    if humor:
        notas = notas + 0.15 * df["texto"].str.contains(humor, case=False).to_numpy()
    if underground:
        notas = notas * np.where(EH_UNDERGROUND, 1.0, 0.5)
    if escopo == "br":
        notas = np.where(EH_BR, notas, -1)
    elif escopo == "int":
        notas = np.where(EH_BR, -1, notas)
    notas = np.array(notas, dtype=float)
    notas[indices] = -1  # não repete o que a pessoa já escolheu

    pesos = perfil.ravel()
    cards = []
    for i in np.argsort(notas)[::-1][:limite]:
        if notas[i] <= 0:
            continue
        comum = np.asarray(MATRIZ[i].todense()).ravel() * pesos
        top = [t for t in VOCAB[np.argsort(comum)[::-1][:3]] if comum[VOCAB == t][0] > 0]
        linha = df.iloc[i]
        busca = quote(f"{linha['nome']} {linha['album']}".strip())
        cards.append({
            "nome": linha["nome"],
            "origem": "Brasil" if linha["pais"] == "BR" else "Internacional",
            "genero": linha["genero"],
            "album": linha["album"],
            "match": int(round(min(notas[i], 1.0) * 100)),
            "porque": top,
            "link": "https://open.spotify.com/search/" + busca,
        })
    return cards, [NOMES[i] for i in indices], nao_achados


# ---------- rotas ----------
@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    humor = request.args.get("humor", "")
    escopo = request.args.get("escopo", "")
    underground = request.args.get("underground") == "1"
    cards, escolhidos, nao_achados = [], [], []
    if q:
        entradas = [p for p in q.replace(";", ",").split(",") if p.strip()]
        cards, escolhidos, nao_achados = recomendar(entradas, humor, escopo, underground)
    return render_template(
        "index.html", q=q, humor=humor, escopo=escopo, underground=underground,
        humores=HUMORES, escopos=ESCOPOS, cards=cards, escolhidos=escolhidos,
        nao_achados=nao_achados, sugestoes=SUGESTOES, buscou=bool(q), total=len(df),
    )


@app.route("/api/sugestoes")
def sugestoes():
    q = normalizar(request.args.get("q", ""))
    if not q:
        return jsonify([])
    achados = process.extract(q, CHAVES, scorer=fuzz.partial_ratio, limit=5, score_cutoff=75)
    return jsonify([NOMES[i] for _, _, i in achados])


@app.route("/api/recomendar")
def api():
    cards, _, _ = recomendar([p for p in request.args.get("q", "").split(",") if p.strip()])
    return jsonify(cards)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
