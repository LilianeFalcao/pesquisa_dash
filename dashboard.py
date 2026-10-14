
import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Dashboard — Consumo e Anomalias",
    page_icon="💧",
    layout="wide"
)

PASTA_DADOS = "./dash"
ARQUIVO_CSV = os.path.join(
    PASTA_DADOS,
    "dashboard_consumo_final.csv"
)

ARQUIVO_BRUTO_2024 = os.path.join(
    PASTA_DADOS, "dados_medidores_2024.csv"
)
ARQUIVO_BRUTO_2025 = os.path.join(
    PASTA_DADOS, "dados_medidores_2025.csv"
)

MESES = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março",
    4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro",
    10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

ORDEM_CONJUNTOS = [
    "Validação 2024",
    "Teste 2024",
    "Aplicação 2025"
]

COLUNA_PREVISAO_GRAFICO = "consumo_previsto_ajustado"


# ============================================================
# FUNÇÕES
# ============================================================

@st.cache_data
def carregar_dados_brutos(caminho, ano):
    """Carrega e padroniza os registros brutos."""
    bruto = pd.read_csv(
        caminho,
        encoding="utf-8-sig",
        low_memory=False
    )

    obrigatorias = [
        "datetime",
        "day_consumption",
        "codigo_medidor"
    ]

    faltantes = [
        coluna for coluna in obrigatorias
        if coluna not in bruto.columns
    ]

    if faltantes:
        raise ValueError(
            f"Colunas ausentes em {os.path.basename(caminho)}: "
            + ", ".join(faltantes)
        )

    bruto["data"] = pd.to_datetime(
        bruto["datetime"],
        errors="coerce"
    )

    bruto["consumo_diario"] = pd.to_numeric(
        bruto["day_consumption"],
        errors="coerce"
    )

    if "sala_nome_completo" in bruto.columns:
        bruto["sala"] = (
            bruto["sala_nome_completo"].astype("string")
        )

    elif {"sala_bloco", "sala_numero"}.issubset(bruto.columns):
        bruto["sala"] = (
            bruto["sala_bloco"].astype("string")
            + " - "
            + bruto["sala_numero"].astype("string")
        )

    else:
        bruto["sala"] = "Sala não identificada"

    bruto["codigo_medidor"] = (
        bruto["codigo_medidor"].astype("string")
    )

    bruto["ano_arquivo"] = ano
    bruto["mes"] = bruto["data"].dt.month
    bruto["mes_nome"] = bruto["mes"].map(MESES)
    bruto["data_dia"] = bruto["data"].dt.date

    bruto["tipo_dia"] = np.where(
        bruto["data"].dt.dayofweek < 5,
        "Dia útil",
        "Fim de semana"
    )

    bruto = bruto.dropna(
        subset=[
            "data",
            "consumo_diario",
            "sala",
            "codigo_medidor"
        ]
    ).copy()

    bruto = bruto[
        bruto["sala"].str.lower().ne("<na>")
        & bruto["sala"].str.strip().ne("")
        & bruto["codigo_medidor"].str.lower().ne("<na>")
        & bruto["codigo_medidor"].str.strip().ne("")
    ].copy()

    # As leituras negativas são preservadas nesta função para
    # auditoria e cálculo da cobertura dos registros.
    bruto["leitura_negativa"] = (
        bruto["consumo_diario"] < 0
    )

    return bruto


@st.cache_data
def carregar_csv(caminho):
    return pd.read_csv(
        caminho,
        encoding="utf-8-sig",
        low_memory=False
    )


def converter_bool(serie):
    """Converte valores booleanos lidos do CSV."""
    if pd.api.types.is_bool_dtype(serie):
        return serie.fillna(False)

    mapa = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "sim": True,
        "não": False,
        "nao": False,
        "yes": True,
        "no": False
    }

    return (
        serie.astype(str)
        .str.strip()
        .str.lower()
        .map(mapa)
        .fillna(False)
        .astype(bool)
    )


def preparar_dataframe(df):
    df = df.copy()

    obrigatorias = [
        "conjunto",
        "sala",
        "codigo_medidor",
        "data_alvo",
        "consumo_real",
        "consumo_previsto",
        "anormal",
        "limiar_erro",
        "erro_absoluto"
    ]

    faltantes = [
        coluna for coluna in obrigatorias
        if coluna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "Colunas ausentes no CSV: "
            + ", ".join(faltantes)
        )

    df["data_alvo"] = pd.to_datetime(
        df["data_alvo"],
        errors="coerce"
    )

    for coluna in ["sala", "codigo_medidor", "conjunto"]:
        df[coluna] = (
            df[coluna]
            .astype("string")
            .fillna("Não informado")
        )

    colunas_numericas = [
        "consumo_real",
        "consumo_previsto",
        "consumo_previsto_ajustado",
        "erro",
        "erro_absoluto",
        "limiar_erro",
        "indice_desvio",
        "quantidade",
        "p95"
    ]

    for coluna in colunas_numericas:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(
                df[coluna],
                errors="coerce"
            )

    df["anormal"] = converter_bool(df["anormal"])

    if "consumo_previsto_ajustado" not in df.columns:
        df["consumo_previsto_ajustado"] = (
            df["consumo_previsto"].clip(lower=0)
        )
    else:
        # Garante que a coluna usada nos gráficos não seja negativa.
        df["consumo_previsto_ajustado"] = (
            df["consumo_previsto_ajustado"].clip(lower=0)
        )

    if "erro" not in df.columns:
        df["erro"] = (
            df["consumo_real"] - df["consumo_previsto"]
        )

    if "erro_absoluto" not in df.columns:
        df["erro_absoluto"] = df["erro"].abs()

    if "previsao_negativa" not in df.columns:
        df["previsao_negativa"] = (
            df["consumo_previsto"] < 0
        )
    else:
        df["previsao_negativa"] = converter_bool(
            df["previsao_negativa"]
        )

    if "tipo_desvio" not in df.columns:
        df["tipo_desvio"] = np.select(
            [
                df["anormal"]
                & (
                    df["consumo_real"]
                    > df["consumo_previsto"]
                ),
                df["anormal"]
                & (
                    df["consumo_real"]
                    < df["consumo_previsto"]
                )
            ],
            [
                "Consumo acima do esperado",
                "Consumo abaixo do esperado"
            ],
            default="Sem anomalia"
        )

    if "fonte_limiar" not in df.columns:
        df["fonte_limiar"] = (
            "Limiar calibrado em 2024"
        )

    df["ano"] = df["data_alvo"].dt.year
    df["mes"] = df["data_alvo"].dt.month
    df["mes_nome"] = df["mes"].map(MESES)

    if "indice_desvio" not in df.columns:
        df["indice_desvio"] = np.where(
            df["limiar_erro"] > 0,
            df["erro_absoluto"] / df["limiar_erro"],
            np.nan
        )

    return df


def calcular_metricas(df):
    """
    Calcula as métricas usando as previsões brutas.
    Viés = previsão bruta - consumo real.
    """
    dados = df.dropna(
        subset=[
            "consumo_real",
            "consumo_previsto"
        ]
    )

    if dados.empty:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "vies": np.nan,
            "negativas": 0,
            "pct_negativas": 0.0
        }

    real = dados["consumo_real"].to_numpy(dtype=float)
    previsto = dados["consumo_previsto"].to_numpy(dtype=float)
    erro = previsto - real

    return {
        "n": len(dados),
        "mae": np.mean(np.abs(erro)),
        "rmse": np.sqrt(np.mean(erro ** 2)),
        "vies": np.mean(erro),
        "negativas": int((previsto < 0).sum()),
        "pct_negativas": 100 * np.mean(previsto < 0)
    }


def formatar_numero(valor, casas=2):
    if pd.isna(valor):
        return "—"

    return (
        f"{valor:,.{casas}f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def nome_normalizado(texto):
    """Normaliza texto para identificar salas sem mapeamento confiável."""
    return (
        str(texto)
        .strip()
        .lower()
        .replace("ã", "a")
        .replace("á", "a")
        .replace("à", "a")
        .replace("â", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("õ", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )


# ============================================================
# CARREGAMENTO DO CSV CONSOLIDADO
# ============================================================

st.title(
    "💧 Consumo de Água — Previsão e Detecção de Anomalias"
)

st.markdown(
    """
    **Modelo:** LSTM para previsão diária do consumo de água.

    **Regra de detecção:** sinaliza observações cujo erro absoluto
    ultrapassa o limiar efetivo calibrado com a validação de 2024.
    O limiar utiliza P95 individual quando há pelo menos 30 observações
    e fallback global nos demais casos, com piso mínimo de 1 L.

    **Importante:** uma anomalia indica um desvio em relação à previsão;
    não confirma a existência de vazamento.
    """
)

if not os.path.exists(ARQUIVO_CSV):
    st.error(
        f"Arquivo não encontrado: `{ARQUIVO_CSV}`. "
        "Coloque o CSV exportado pelo notebook nessa pasta."
    )
    st.stop()

try:
    df_todos = preparar_dataframe(
        carregar_csv(ARQUIVO_CSV)
    )
except Exception as erro:
    st.error(f"Erro ao carregar o CSV: {erro}")
    st.stop()


# ============================================================
# AUDITORIA DA CLASSIFICAÇÃO
# ============================================================

with st.expander(
    "🔎 Auditoria da classificação",
    expanded=False
):
    auditoria = df_todos.copy()

    auditoria["erro_calculado"] = (
        auditoria["consumo_real"]
        - auditoria["consumo_previsto"]
    )

    auditoria["erro_abs_calculado"] = (
        auditoria["erro_calculado"].abs()
    )

    validos = auditoria[
        auditoria["erro_abs_calculado"].notna()
        & auditoria["limiar_erro"].notna()
        & (auditoria["limiar_erro"] >= 0)
    ].copy()

    validos["anormal_calculado"] = (
        validos["erro_abs_calculado"]
        > validos["limiar_erro"]
    )

    validos["divergencia_classificacao"] = (
        validos["anormal"]
        != validos["anormal_calculado"]
    )

    validos["divergencia_erro"] = ~np.isclose(
        validos["erro_absoluto"],
        validos["erro_abs_calculado"],
        rtol=1e-5,
        atol=1e-5
    )

    total = len(validos)
    divergencias = int(
        validos["divergencia_classificacao"].sum()
    )
    erros_divergentes = int(
        validos["divergencia_erro"].sum()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Registros auditáveis",
        f"{total:,}".replace(",", ".")
    )
    c2.metric(
        "Classificações divergentes",
        f"{divergencias:,}".replace(",", ".")
    )
    c3.metric(
        "Erros absolutos divergentes",
        f"{erros_divergentes:,}".replace(",", ".")
    )

    st.caption(
        "Regra conferida: uma observação é anormal quando "
        "o erro absoluto é maior que o limiar efetivo."
    )

    if divergencias > 0:
        st.warning(
            "Existem classificações que não correspondem à "
            "regra recalculada. Confira os registros antes "
            "de corrigir o CSV."
        )

        st.dataframe(
            validos.loc[
                validos["divergencia_classificacao"],
                [
                    "conjunto",
                    "sala",
                    "codigo_medidor",
                    "data_alvo",
                    "consumo_real",
                    "consumo_previsto",
                    "erro_absoluto",
                    "limiar_erro",
                    "anormal",
                    "anormal_calculado"
                ]
            ].head(100),
            width="stretch",
            hide_index=True
        )
    else:
        st.success(
            "As classificações conferem com a regra para "
            "todos os registros auditáveis."
        )

    if erros_divergentes > 0:
        st.warning(
            f"{erros_divergentes:,} registros têm erro absoluto "
            "diferente do valor recalculado.".replace(",", ".")
        )

    st.subheader("Distribuição dos limiares por conjunto")

    resumo_limiares = (
        df_todos.groupby("conjunto")["limiar_erro"]
        .agg(
            registros="count",
            mediana="median",
            minimo="min",
            maximo="max"
        )
        .reset_index()
    )

    st.dataframe(
        resumo_limiares,
        width="stretch",
        hide_index=True
    )

    st.subheader("Origem dos limiares")

    if "fonte_limiar" in df_todos.columns:
        st.dataframe(
            pd.crosstab(
                df_todos["conjunto"],
                df_todos["fonte_limiar"],
                margins=True
            ),
            width="stretch"
        )


# ============================================================
# CONJUNTOS DISPONÍVEIS
# ============================================================

conjuntos_disponiveis = (
    df_todos["conjunto"].dropna().unique().tolist()
)

opcoes = [
    nome for nome in ORDEM_CONJUNTOS
    if nome in conjuntos_disponiveis
]

opcoes += sorted([
    nome for nome in conjuntos_disponiveis
    if nome not in opcoes
])

if not opcoes:
    st.error("Nenhum conjunto de dados foi encontrado no CSV.")
    st.stop()


# ============================================================
# SELEÇÃO DO CONJUNTO
# ============================================================

st.sidebar.header("📁 Conjunto de dados")

indice_padrao = (
    opcoes.index("Teste 2024")
    if "Teste 2024" in opcoes
    else 0
)

conjunto = st.sidebar.selectbox(
    "Conjunto",
    opcoes,
    index=indice_padrao
)

df = df_todos[
    df_todos["conjunto"] == conjunto
].copy()


# ============================================================
# FILTROS
# ============================================================

st.sidebar.header("🔎 Filtros")
df_filtrado = df.copy()

anos = sorted(
    df_filtrado["ano"].dropna().astype(int).unique()
)

anos_sel = st.sidebar.multiselect(
    "Ano",
    anos,
    default=anos
)

df_filtrado = df_filtrado[
    df_filtrado["ano"].isin(anos_sel)
]

meses = sorted(
    df_filtrado["mes"].dropna().astype(int).unique()
)

meses_sel = st.sidebar.multiselect(
    "Mês",
    meses,
    default=meses,
    format_func=lambda x: MESES.get(int(x), str(x))
)

df_filtrado = df_filtrado[
    df_filtrado["mes"].isin(meses_sel)
]

salas = sorted(
    df_filtrado["sala"].dropna().astype(str).unique()
)

sala_sel = st.sidebar.selectbox(
    "Sala",
    ["Todas as salas"] + salas
)

if sala_sel != "Todas as salas":
    df_filtrado = df_filtrado[
        df_filtrado["sala"] == sala_sel
    ]

medidores = sorted(
    df_filtrado["codigo_medidor"]
    .dropna()
    .astype(str)
    .unique()
)

medidores_sel = st.sidebar.multiselect(
    "Medidor (opcional)",
    medidores
)

if medidores_sel:
    df_filtrado = df_filtrado[
        df_filtrado["codigo_medidor"].isin(medidores_sel)
    ]

datas_validas = df_filtrado["data_alvo"].dropna()

if not datas_validas.empty:
    data_min = datas_validas.min().date()
    data_max = datas_validas.max().date()

    periodo = st.sidebar.date_input(
        "Período",
        value=(data_min, data_max)
    )

    if isinstance(periodo, (tuple, list)) and len(periodo) == 2:
        inicio = pd.Timestamp(periodo[0])
        fim = pd.Timestamp(periodo[1]) + pd.Timedelta(days=1)

        df_filtrado = df_filtrado[
            (df_filtrado["data_alvo"] >= inicio)
            & (df_filtrado["data_alvo"] < fim)
        ]

if st.sidebar.checkbox("Somente observações anormais"):
    df_filtrado = df_filtrado[
        df_filtrado["anormal"]
    ]

tipos = sorted(
    df_filtrado["tipo_desvio"].dropna().unique()
)

tipos_sel = st.sidebar.multiselect(
    "Tipo de desvio (opcional)",
    tipos
)

if tipos_sel:
    df_filtrado = df_filtrado[
        df_filtrado["tipo_desvio"].isin(tipos_sel)
    ]

if st.sidebar.checkbox("Somente previsões brutas negativas"):
    df_filtrado = df_filtrado[
        df_filtrado["previsao_negativa"]
    ]

st.sidebar.divider()
st.sidebar.caption(
    f"Registros após filtros: {len(df_filtrado):,}"
    .replace(",", ".")
)


# ============================================================
# INDICADORES PRINCIPAIS
# ============================================================

st.subheader(conjunto)

metricas = calcular_metricas(df_filtrado)
n = len(df_filtrado)

n_anomalias = (
    int(df_filtrado["anormal"].sum()) if n else 0
)

taxa_anomalias = (
    100 * n_anomalias / n if n else 0
)

n_zero_anomalias = int(
    (
        df_filtrado["anormal"]
        & (df_filtrado["consumo_real"] == 0)
    ).sum()
) if n else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric(
    "Observações",
    f"{n:,}".replace(",", ".")
)
c2.metric(
    "Anomalias",
    f"{n_anomalias:,}".replace(",", ".")
)
c3.metric(
    "Taxa de anomalias",
    f"{formatar_numero(taxa_anomalias)}%"
)
c4.metric(
    "MAE bruto",
    f"{formatar_numero(metricas['mae'])} L"
)
c5.metric(
    "RMSE bruto",
    f"{formatar_numero(metricas['rmse'])} L"
)
c6.metric(
    "Viés bruto",
    f"{formatar_numero(metricas['vies'])} L"
)

c7, c8, c9 = st.columns(3)

c7.metric(
    "Medidores",
    f"{df_filtrado[['sala', 'codigo_medidor']].drop_duplicates().shape[0]:,}"
    .replace(",", ".")
)
c8.metric(
    "Previsões negativas",
    f"{metricas['negativas']:,}".replace(",", ".")
)
c9.metric(
    "Anomalias com consumo zero",
    f"{n_zero_anomalias:,}".replace(",", ".")
)

st.caption(
    "Viés = previsão bruta − consumo real. "
    "Valores negativos indicam subestimação média. "
    "As métricas são recalculadas conforme os filtros ativos."
)


# ============================================================
# ABAS
# ============================================================

(
    aba_visao,
    aba_perfis,
    aba_cobertura,
    aba_anomalias,
    aba_erros,
    aba_series,
    aba_dados
) = st.tabs([
    "📊 Visão geral",
    "🏢 Perfil de consumo",
    "📅 Cobertura e amostras",
    "🚨 Anomalias",
    "📉 Erros",
    "📈 Série temporal",
    "📋 Dados"
])


# ============================================================
# COBERTURA DOS DADOS E DISTRIBUIÇÃO DAS AMOSTRAS
# ============================================================

with aba_cobertura:
    st.header("📅 Cobertura dos dados e formação das amostras")

    st.markdown(
        """
        Esta seção apresenta duas perspectivas distintas:
        **a cobertura dos registros brutos por medidor** e
        **a distribuição das observações utilizadas na avaliação da LSTM**.
        """
    )

    # --------------------------------------------------------
    # 1. COBERTURA DOS REGISTROS BRUTOS
    # --------------------------------------------------------

    st.subheader("1. Cobertura dos registros por ano")

    arquivos_cobertura = []

    if os.path.exists(ARQUIVO_BRUTO_2024):
        arquivos_cobertura.append(
            (ARQUIVO_BRUTO_2024, 2024)
        )

    if os.path.exists(ARQUIVO_BRUTO_2025):
        arquivos_cobertura.append(
            (ARQUIVO_BRUTO_2025, 2025)
        )

    partes_cobertura = []

    for caminho, ano_ref in arquivos_cobertura:
        try:
            parte = carregar_dados_brutos(
                caminho,
                ano_ref
            )
            partes_cobertura.append(parte)

        except Exception as exc:
            st.error(
                f"Não foi possível carregar os dados de "
                f"{ano_ref}: {exc}"
            )

    if partes_cobertura:
        cobertura = pd.concat(
            partes_cobertura,
            ignore_index=True
        )

        # Identificação única por ano, sala e medidor.
        dias_medidor = (
            cobertura.groupby(
                ["ano_arquivo", "sala", "codigo_medidor"],
                as_index=False
            )
            .agg(
                dias_registrados=("data_dia", "nunique"),
                primeira_data=("data", "min"),
                ultima_data=("data", "max")
            )
        )

        resumo_cobertura = (
            dias_medidor.groupby(
                "ano_arquivo",
                as_index=False
            )
            .agg(
                medidores=("codigo_medidor", "size"),
                media_dias=("dias_registrados", "mean"),
                mediana_dias=("dias_registrados", "median"),
                minimo_dias=("dias_registrados", "min"),
                maximo_dias=("dias_registrados", "max")
            )
        )

        colunas_ano = st.columns(
            max(1, len(resumo_cobertura))
        )

        for coluna, (_, linha) in zip(
            colunas_ano,
            resumo_cobertura.iterrows()
        ):
            ano_ref = int(linha["ano_arquivo"])

            with coluna:
                st.markdown(f"### {ano_ref}")

                st.metric(
                    "Medidores/sala com registros",
                    f"{int(linha['medidores']):,}".replace(",", ".")
                )

                st.metric(
                    "Média de dias registrados",
                    formatar_numero(linha["media_dias"], 1)
                )

                st.metric(
                    "Mediana de dias registrados",
                    formatar_numero(linha["mediana_dias"], 0)
                )

                st.caption(
                    f"Intervalo: {int(linha['minimo_dias'])} a "
                    f"{int(linha['maximo_dias'])} dias distintos."
                )

        fig_cobertura = px.box(
            dias_medidor,
            x="ano_arquivo",
            y="dias_registrados",
            points="all",
            color="ano_arquivo",
            labels={
                "ano_arquivo": "Ano",
                "dias_registrados": "Dias distintos registrados"
            },
            title="Distribuição da cobertura entre medidores"
        )

        st.plotly_chart(
            fig_cobertura,
            width="stretch"
        )

        st.caption(
            "Cada ponto representa uma combinação de sala e medidor "
            "em um ano. A cobertura considera datas distintas com "
            "registros válidos segundo os critérios de carregamento."
        )

        # ----------------------------------------------------
        # 2. COBERTURA MENSAL POR MEDIDOR
        # ----------------------------------------------------

        st.subheader("2. Cobertura mensal dos registros")

        cobertura_mensal = cobertura.copy()

        cobertura_mensal["mes_data"] = (
            cobertura_mensal["data"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        dias_por_medidor_mes = (
            cobertura_mensal.groupby(
                [
                    "ano_arquivo",
                    "sala",
                    "codigo_medidor",
                    "mes_data"
                ],
                as_index=False
            )
            .agg(
                dias_registrados=("data_dia", "nunique")
            )
        )

        dias_por_medidor_mes["dias_calendario"] = (
            dias_por_medidor_mes["mes_data"].dt.days_in_month
        )

        dias_por_medidor_mes["cobertura_pct"] = (
            100
            * dias_por_medidor_mes["dias_registrados"]
            / dias_por_medidor_mes["dias_calendario"]
        )

        anos_heat = sorted(
            dias_por_medidor_mes["ano_arquivo"]
            .dropna()
            .unique()
        )

        if len(anos_heat):
            ano_heat = st.selectbox(
                "Ano para visualizar a cobertura mensal",
                anos_heat,
                key="ano_heat_cobertura"
            )

            dados_heat = dias_por_medidor_mes[
                dias_por_medidor_mes["ano_arquivo"] == ano_heat
            ].copy()

            dados_heat["medidor_label"] = (
                dados_heat["sala"].astype(str)
                + " | "
                + dados_heat["codigo_medidor"].astype(str)
            )

            ordem_medidores = (
                dados_heat.groupby("medidor_label")["cobertura_pct"]
                .mean()
                .sort_values()
                .index.tolist()
            )

            if ordem_medidores:
                limite_maximo = min(
                    100,
                    len(ordem_medidores)
                )
                limite_padrao = min(
                    30,
                    limite_maximo
                )

                limite_medidores = st.slider(
                    "Quantidade de medidores no mapa de calor",
                    min_value=1,
                    max_value=limite_maximo,
                    value=limite_padrao,
                    key="limite_heat_cobertura"
                )

                medidores_heat = ordem_medidores[
                    :limite_medidores
                ]

                heat = dados_heat[
                    dados_heat["medidor_label"].isin(
                        medidores_heat
                    )
                ].copy()

                matriz_cobertura = heat.pivot(
                    index="medidor_label",
                    columns="mes_data",
                    values="cobertura_pct"
                )

                matriz_cobertura = matriz_cobertura.reindex(
                    index=medidores_heat
                )

                meses_ano = pd.date_range(
                    start=f"{int(ano_heat)}-01-01",
                    end=f"{int(ano_heat)}-12-01",
                    freq="MS"
                )

                matriz_cobertura = matriz_cobertura.reindex(
                    columns=meses_ano
                )

                matriz_cobertura.columns = [
                    MESES[data.month]
                    for data in matriz_cobertura.columns
                ]

                fig_heat_cobertura = px.imshow(
                    matriz_cobertura,
                    aspect="auto",
                    color_continuous_scale="RdYlGn",
                    zmin=0,
                    zmax=100,
                    labels={
                        "x": "Mês",
                        "y": "Sala | medidor",
                        "color": "Cobertura (%)"
                    },
                    title=(
                        "Percentual de dias com registro por "
                        f"medidor — {ano_heat}"
                    )
                )

                fig_heat_cobertura.update_layout(
                    height=max(
                        420,
                        min(
                            1000,
                            25 * len(matriz_cobertura) + 180
                        )
                    ),
                    margin={
                        "l": 20,
                        "r": 20,
                        "t": 70,
                        "b": 40
                    }
                )

                fig_heat_cobertura.update_traces(
                    hovertemplate=(
                        "Sala/medidor: %{y}<br>"
                        "Mês: %{x}<br>"
                        "Cobertura: %{z:.1f}%<extra></extra>"
                    )
                )

                st.plotly_chart(
                    fig_heat_cobertura,
                    width="stretch"
                )

                st.caption(
                    "Cobertura mensal = dias distintos com registro "
                    "divididos pelo número de dias do mês. Células "
                    "sem cor indicam ausência de registros naquele "
                    "mês, não consumo igual a zero."
                )
            else:
                st.info(
                    "Não há medidores disponíveis para o mapa de calor."
                )

    else:
        st.info(
            "Adicione os arquivos brutos de 2024 e/ou 2025 à pasta "
            "`dash/` para visualizar a cobertura dos registros."
        )

    # --------------------------------------------------------
    # 3. DISTRIBUIÇÃO MENSAL DAS AMOSTRAS DA LSTM
    # --------------------------------------------------------

    st.divider()
    st.subheader("3. Distribuição temporal das amostras avaliadas")

    dados_amostras = df_todos.dropna(
        subset=["data_alvo", "conjunto"]
    ).copy()

    dados_amostras["mes_data"] = (
        dados_amostras["data_alvo"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    resumo_amostras = (
        dados_amostras.groupby(
            ["conjunto", "mes_data"],
            as_index=False
        )
        .agg(
            observacoes=("data_alvo", "size"),
            medidores=("codigo_medidor", "nunique")
        )
    )

    ordem_conjuntos_presentes = [
        nome for nome in ORDEM_CONJUNTOS
        if nome in resumo_amostras["conjunto"].unique()
    ]

    if not resumo_amostras.empty:
        fig_amostras = px.line(
            resumo_amostras,
            x="mes_data",
            y="observacoes",
            color="conjunto",
            markers=True,
            category_orders={
                "conjunto": ordem_conjuntos_presentes
            },
            labels={
                "mes_data": "Mês de referência da previsão",
                "observacoes": "Quantidade de previsões",
                "conjunto": "Conjunto"
            },
            title="Quantidade de previsões por mês e conjunto"
        )

        fig_amostras.update_layout(
            xaxis=dict(
                tickformat="%b/%Y",
                dtick="M1"
            ),
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_amostras,
            width="stretch"
        )
    else:
        st.info(
            "Não há datas válidas para exibir a distribuição mensal."
        )

    resumo_conjuntos = (
        dados_amostras.groupby("conjunto", as_index=False)
        .agg(
            observacoes=("data_alvo", "size"),
            medidores=("codigo_medidor", "nunique"),
            primeira_previsao=("data_alvo", "min"),
            ultima_previsao=("data_alvo", "max")
        )
    )

    if not resumo_conjuntos.empty:
        resumo_conjuntos["periodo"] = (
            resumo_conjuntos["primeira_previsao"].dt.strftime("%d/%m/%Y")
            + " a "
            + resumo_conjuntos["ultima_previsao"].dt.strftime("%d/%m/%Y")
        )

        resumo_conjuntos = resumo_conjuntos[
            [
                "conjunto",
                "observacoes",
                "medidores",
                "periodo"
            ]
        ].rename(
            columns={
                "conjunto": "Conjunto",
                "observacoes": "Quantidade de previsões",
                "medidores": "Medidores",
                "periodo": "Intervalo entre primeira e última previsão"
            }
        )

        st.dataframe(
            resumo_conjuntos,
            width="stretch",
            hide_index=True
        )

    st.markdown(
        """
        **Como interpretar a distribuição:**

        - **Validação 2024:** amostras usadas para avaliar o modelo e
          calibrar os limiares de desvio.
        - **Teste 2024:** amostras separadas para avaliar o desempenho
          em observações não utilizadas no treinamento nem na validação.
        - **Aplicação 2025:** previsões realizadas sobre os registros
          desse ano, conforme a disponibilidade de janelas válidas.

        A divisão de 2024 é cronológica dentro de cada medidor.
        Por isso, as datas dos conjuntos podem se sobrepor quando
        considerados todos os medidores em conjunto. As quantidades
        dependem da disponibilidade de sequências válidas de oito dias:
        sete dias de entrada e um dia-alvo.
        """
    )


# ============================================================
# VISÃO GERAL
# ============================================================

with aba_visao:
    if df_filtrado.empty:
        st.warning("Nenhum registro corresponde aos filtros.")
    else:
        st.subheader("Consumo real × consumo previsto")

        dados_tempo = (
            df_filtrado.groupby("data_alvo", as_index=False)
            .agg(
                consumo_real=("consumo_real", "mean"),
                consumo_previsto=(
                    COLUNA_PREVISAO_GRAFICO,
                    "mean"
                )
            )
            .sort_values("data_alvo")
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dados_tempo["data_alvo"],
            y=dados_tempo["consumo_real"],
            mode="lines",
            name="Consumo real"
        ))

        fig.add_trace(go.Scatter(
            x=dados_tempo["data_alvo"],
            y=dados_tempo["consumo_previsto"],
            mode="lines",
            name="Consumo previsto ajustado"
        ))

        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="Consumo médio (L)",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.subheader("Distribuição das classificações")

        resumo = (
            df_filtrado["anormal"]
            .map({False: "Normal", True: "Anormal"})
            .value_counts()
            .rename_axis("Classificação")
            .reset_index(name="Quantidade")
        )

        fig_class = px.bar(
            resumo,
            x="Classificação",
            y="Quantidade",
            text_auto=True
        )

        st.plotly_chart(
            fig_class,
            width="stretch"
        )

        st.subheader("Consumo real × previsão ajustada")

        fig_scatter = px.scatter(
            df_filtrado,
            x="consumo_real",
            y=COLUNA_PREVISAO_GRAFICO,
            color=df_filtrado["anormal"].map({
                False: "Normal",
                True: "Anormal"
            }),
            hover_data=[
                "sala",
                "codigo_medidor",
                "data_alvo"
            ],
            labels={
                "consumo_real": "Consumo real (L)",
                COLUNA_PREVISAO_GRAFICO:
                    "Consumo previsto ajustado (L)",
                "color": "Classificação"
            }
        )

        st.plotly_chart(
            fig_scatter,
            width="stretch"
        )

        st.info(
            "As previsões exibidas nos gráficos são limitadas a zero. "
            "As métricas oficiais e a classificação de anomalias usam "
            "as previsões brutas."
        )


# ============================================================
# PERFIL DE CONSUMO
# ============================================================

with aba_perfis:
    st.header("🏢 Caracterização dos perfis de consumo")

    st.markdown(
        """
        Esta seção utiliza os registros brutos de consumo diário
        para comparar os períodos e caracterizar as salas.
        Os gráficos descritivos não usam as previsões da LSTM.

        Leituras negativas são preservadas na auditoria da cobertura,
        mas excluídas dos indicadores descritivos de consumo.
        """
    )

    arquivos_brutos = []

    if os.path.exists(ARQUIVO_BRUTO_2024):
        arquivos_brutos.append(
            (ARQUIVO_BRUTO_2024, 2024)
        )

    if os.path.exists(ARQUIVO_BRUTO_2025):
        arquivos_brutos.append(
            (ARQUIVO_BRUTO_2025, 2025)
        )

    if not arquivos_brutos:
        st.warning(
            "Para habilitar esta seção, coloque os arquivos "
            "`dados_medidores_2024.csv` e `dados_medidores_2025.csv` "
            "na pasta `dash/` do repositório."
        )

    else:
        partes = []
        erros_carga = []

        for caminho_bruto, ano_bruto in arquivos_brutos:
            try:
                partes.append(
                    carregar_dados_brutos(
                        caminho_bruto,
                        ano_bruto
                    )
                )
            except Exception as exc:
                erros_carga.append(
                    f"{os.path.basename(caminho_bruto)}: {exc}"
                )

        if erros_carga:
            st.warning(
                "Alguns arquivos brutos não puderam ser carregados: "
                + " | ".join(erros_carga)
            )

        if not partes:
            st.error(
                "Não foi possível carregar os arquivos brutos "
                "para a análise de perfil."
            )

        else:
            perfil = pd.concat(
                partes,
                ignore_index=True
            )

            # As leituras negativas são excluídas apenas da análise
            # descritiva, não da auditoria da cobertura.
            perfil = perfil[
                perfil["consumo_diario"] >= 0
            ].copy()

            anos_perfil = sorted(
                perfil["ano_arquivo"].dropna().unique().tolist()
            )

            col_filtros = st.columns([1, 1, 2])

            with col_filtros[0]:
                anos_perfil_sel = st.multiselect(
                    "Ano(s) para analisar",
                    anos_perfil,
                    default=anos_perfil,
                    key="perfil_anos"
                )

            perfil = perfil[
                perfil["ano_arquivo"].isin(anos_perfil_sel)
            ].copy()

            with col_filtros[1]:
                meses_perfil_sel = st.multiselect(
                    "Meses",
                    list(range(1, 13)),
                    default=list(range(1, 13)),
                    format_func=lambda x: MESES[x],
                    key="perfil_meses"
                )

            perfil = perfil[
                perfil["mes"].isin(meses_perfil_sel)
            ].copy()

            salas_perfil = sorted(
                perfil["sala"].dropna().unique().tolist()
            )

            with col_filtros[2]:
                salas_perfil_sel = st.multiselect(
                    "Salas (opcional)",
                    salas_perfil,
                    key="perfil_salas"
                )

            if salas_perfil_sel:
                perfil = perfil[
                    perfil["sala"].isin(salas_perfil_sel)
                ].copy()

            if perfil.empty:
                st.info(
                    "Não há registros para os filtros selecionados."
                )

            else:
                mensal = (
                    perfil.assign(
                        mes_data=(
                            perfil["data"]
                            .dt.to_period("M")
                            .dt.to_timestamp()
                        )
                    )
                    .groupby(
                        ["ano_arquivo", "mes_data"],
                        as_index=False
                    )
                    .agg(
                        consumo_total_l=("consumo_diario", "sum"),
                        consumo_medio_registro_l=("consumo_diario", "mean"),
                        dias_registrados=("data_dia", "nunique"),
                        medidores=("codigo_medidor", "nunique")
                    )
                )

                mensal["consumo_medio_diario_l"] = (
                    mensal["consumo_total_l"]
                    / mensal["mes_data"].dt.days_in_month
                )

                total_l = perfil["consumo_diario"].sum()
                media_registro = perfil["consumo_diario"].mean()
                mediana_registro = perfil["consumo_diario"].median()
                salas_n = perfil["sala"].nunique()

                k1, k2, k3, k4 = st.columns(4)

                k1.metric(
                    "Consumo acumulado nos filtros",
                    f"{formatar_numero(total_l, 1)} L"
                )
                k2.metric(
                    "Média por registro",
                    f"{formatar_numero(media_registro, 2)} L"
                )
                k3.metric(
                    "Mediana por registro",
                    f"{formatar_numero(mediana_registro, 2)} L"
                )
                k4.metric(
                    "Salas analisadas",
                    f"{salas_n:,}".replace(",", ".")
                )

                # ------------------------------------------------
                # 1. DIAS ÚTEIS X FINAIS DE SEMANA
                # ------------------------------------------------

                st.subheader(
                    "1. Comparação entre dias úteis e finais de semana"
                )

                diario_completo = (
                    perfil.groupby(
                        ["data_dia", "tipo_dia"],
                        as_index=False
                    )
                    .agg(
                        consumo_total_dia_l=(
                            "consumo_diario",
                            "sum"
                        )
                    )
                )

                comparacao_dias = (
                    diario_completo.groupby(
                        "tipo_dia",
                        as_index=False
                    )
                    .agg(
                        media_consumo_diario_l=(
                            "consumo_total_dia_l",
                            "mean"
                        ),
                        mediana_consumo_diario_l=(
                            "consumo_total_dia_l",
                            "median"
                        ),
                        dias_observados=(
                            "data_dia",
                            "nunique"
                        )
                    )
                )

                ordem_tipos = [
                    "Dia útil",
                    "Fim de semana"
                ]

                comparacao_dias["tipo_dia"] = pd.Categorical(
                    comparacao_dias["tipo_dia"],
                    categories=ordem_tipos,
                    ordered=True
                )

                comparacao_dias = comparacao_dias.sort_values(
                    "tipo_dia"
                )

                fig_dias = px.bar(
                    comparacao_dias,
                    x="tipo_dia",
                    y="media_consumo_diario_l",
                    labels={
                        "tipo_dia": "Tipo de dia",
                        "media_consumo_diario_l":
                            "Consumo médio diário total (L)"
                    },
                    title=(
                        "Média do consumo diário total: "
                        "dias úteis × finais de semana"
                    ),
                    hover_data={
                        "mediana_consumo_diario_l": ":.2f",
                        "dias_observados": True
                    }
                )

                st.plotly_chart(
                    fig_dias,
                    width="stretch"
                )

                st.caption(
                    "Dias úteis são considerados de segunda a sexta-feira; "
                    "finais de semana, sábado e domingo. Feriados não são "
                    "classificados separadamente."
                )

                # ------------------------------------------------
                # 2. EVOLUÇÃO MENSAL
                # ------------------------------------------------

                st.subheader(
                    "2. Evolução do consumo mensal total"
                )

                fig_mensal = px.line(
                    mensal,
                    x="mes_data",
                    y="consumo_total_l",
                    color="ano_arquivo",
                    markers=True,
                    labels={
                        "mes_data": "Mês",
                        "consumo_total_l": "Consumo total (L)",
                        "ano_arquivo": "Ano"
                    },
                    title="Consumo total registrado por mês"
                )

                fig_mensal.update_layout(
                    hovermode="x unified"
                )

                st.plotly_chart(
                    fig_mensal,
                    width="stretch"
                )

                # ------------------------------------------------
                # 3. COMPARAÇÃO ENTRE ANOS
                # ------------------------------------------------

                st.subheader(
                    "3. Comparação entre anos por mês do calendário"
                )

                mensal["mes_numero"] = (
                    mensal["mes_data"].dt.month
                )

                comparacao = (
                    mensal.groupby(
                        ["ano_arquivo", "mes_numero"],
                        as_index=False
                    )
                    .agg(
                        consumo_total_l=("consumo_total_l", "sum")
                    )
                )

                comparacao["mes_nome"] = (
                    comparacao["mes_numero"].map(MESES)
                )

                fig_comp = px.bar(
                    comparacao,
                    x="mes_nome",
                    y="consumo_total_l",
                    color="ano_arquivo",
                    barmode="group",
                    category_orders={
                        "mes_nome": list(MESES.values())
                    },
                    labels={
                        "mes_nome": "Mês",
                        "consumo_total_l": "Consumo total (L)",
                        "ano_arquivo": "Ano"
                    },
                    title="Comparação do consumo mensal entre anos"
                )

                st.plotly_chart(
                    fig_comp,
                    width="stretch"
                )

                st.caption(
                    "Compare meses equivalentes. Se um ano tiver registros "
                    "incompletos, a diferença pode refletir também a "
                    "cobertura dos dados."
                )

                # ------------------------------------------------
                # 4. CONSUMO MÉDIO DIÁRIO POR SALA
                # ------------------------------------------------

                st.subheader(
                    "4. Consumo médio diário por sala"
                )

                diario_por_sala_ano = (
                    perfil.groupby(
                        ["sala", "ano_arquivo", "data_dia"],
                        as_index=False
                    )
                    .agg(
                        consumo_total_dia_sala_l=(
                            "consumo_diario",
                            "sum"
                        ),
                        medidores_no_dia=(
                            "codigo_medidor",
                            "nunique"
                        )
                    )
                )

                por_sala = (
                    diario_por_sala_ano.groupby(
                        ["sala", "ano_arquivo"],
                        as_index=False
                    )
                    .agg(
                        consumo_medio_diario_l=(
                            "consumo_total_dia_sala_l",
                            "mean"
                        ),
                        consumo_mediano_diario_l=(
                            "consumo_total_dia_sala_l",
                            "median"
                        ),
                        dias_observados=("data_dia", "nunique"),
                        medidores=("medidores_no_dia", "max")
                    )
                )

                sala_norm = (
                    por_sala["sala"]
                    .astype(str)
                    .map(nome_normalizado)
                )

                mask_sala_valida = ~sala_norm.str.contains(
                    r"nao mapead|nao identificad|sem sala|desconhecid",
                    regex=True,
                    na=False
                )

                por_sala_mapeadas = por_sala[
                    mask_sala_valida
                ].copy()

                if (~mask_sala_valida).any():
                    st.caption(
                        "Salas sem identificação confiável foram excluídas "
                        "apenas dos gráficos comparativos por sala; seus "
                        "registros permanecem nos indicadores gerais."
                    )

                if not por_sala_mapeadas.empty:
                    top_n = st.slider(
                        "Quantidade de salas no gráfico",
                        min_value=1,
                        max_value=min(40, len(por_sala_mapeadas["sala"].unique())),
                        value=min(15, len(por_sala_mapeadas["sala"].unique())),
                        key="perfil_top_n"
                    )

                    ordenar_ano = (
                        anos_perfil_sel[-1]
                        if anos_perfil_sel
                        else int(perfil["ano_arquivo"].max())
                    )

                    ranking = (
                        por_sala_mapeadas[
                            por_sala_mapeadas["ano_arquivo"] == ordenar_ano
                        ]
                        .sort_values(
                            "consumo_medio_diario_l",
                            ascending=False
                        )
                        .head(top_n)
                    )

                    if not ranking.empty:
                        ranking_plot = ranking.sort_values(
                            "consumo_medio_diario_l"
                        ).copy()

                        ranking_plot["consumo_plot"] = (
                            ranking_plot["consumo_medio_diario_l"]
                        )

                        max_valor = float(
                            ranking_plot["consumo_plot"].max()
                        )

                        fig_salas = px.bar(
                            ranking_plot,
                            x="consumo_plot",
                            y="sala",
                            orientation="h",
                            hover_data={
                                "consumo_plot": ":.2f",
                                "consumo_medio_diario_l": ":.2f",
                                "consumo_mediano_diario_l": ":.2f",
                                "dias_observados": True,
                                "medidores": True
                            },
                            labels={
                                "consumo_plot": "Consumo médio diário (L/dia)",
                                "consumo_medio_diario_l": "Consumo médio diário (L/dia)",
                                "consumo_mediano_diario_l": "Consumo mediano diário (L/dia)",
                                "sala": "Sala"
                            },
                            title=(
                                f"{len(ranking_plot)} salas com maior "
                                f"consumo médio diário — {ordenar_ano}"
                            )
                        )

                        if max_valor > 0:
                            passo_bruto = max_valor / 5
                            magnitude = (
                                10 ** math.floor(math.log10(passo_bruto))
                                if passo_bruto > 0
                                else 1
                            )

                            candidatos = [1, 2, 2.5, 5, 10]

                            passo = next(
                                (
                                    c * magnitude
                                    for c in candidatos
                                    if c * magnitude >= passo_bruto
                                ),
                                10 * magnitude
                            )

                            fig_salas.update_xaxes(
                                range=[0, max_valor * 1.15],
                                dtick=passo,
                                tick0=0,
                                title="Consumo médio diário (L/dia)"
                            )

                        fig_salas.update_layout(
                            margin={
                                "l": 20,
                                "r": 20,
                                "t": 70,
                                "b": 45
                            }
                        )

                        st.plotly_chart(
                            fig_salas,
                            width="stretch"
                        )

                    st.dataframe(
                        por_sala_mapeadas.sort_values(
                            [
                                "ano_arquivo",
                                "consumo_medio_diario_l"
                            ],
                            ascending=[True, False]
                        ),
                        width="stretch",
                        hide_index=True
                    )

                else:
                    st.info(
                        "Não há salas com identificação confiável "
                        "para compor o ranking."
                    )

                # ------------------------------------------------
                # 5. MAPA DE CALOR
                # ------------------------------------------------

                st.subheader(
                    "5. Mapa de calor: sala × mês"
                )

                perfil_heat = perfil.copy()
                perfil_heat["mes_nome"] = (
                    perfil_heat["mes"].map(MESES)
                )

                heat_diario = (
                    perfil_heat.groupby(
                        ["sala", "mes_nome", "data_dia"],
                        as_index=False
                    )
                    .agg(
                        total_diario_sala_l=(
                            "consumo_diario",
                            "sum"
                        )
                    )
                )

                heat = (
                    heat_diario.groupby(
                        ["sala", "mes_nome"],
                        as_index=False
                    )
                    .agg(
                        media_l=("total_diario_sala_l", "mean")
                    )
                )

                heat_pivot = heat.pivot(
                    index="sala",
                    columns="mes_nome",
                    values="media_l"
                )

                heat_pivot = heat_pivot.reindex(
                    columns=[
                        mes for mes in MESES.values()
                        if mes in heat_pivot.columns
                    ]
                )

                salas_heat = (
                    por_sala_mapeadas
                    .sort_values(
                        "consumo_medio_diario_l",
                        ascending=False
                    )["sala"]
                    .drop_duplicates()
                    .head(30)
                    .tolist()
                )

                heat_pivot = heat_pivot.loc[
                    heat_pivot.index.intersection(salas_heat)
                ]

                if not heat_pivot.empty:
                    valores_validos = heat_pivot.to_numpy(
                        dtype=float
                    )

                    valores_validos = valores_validos[
                        np.isfinite(valores_validos)
                    ]

                    valores_positivos = valores_validos[
                        valores_validos > 0
                    ]

                    limite_cor = (
                        float(np.percentile(valores_positivos, 95))
                        if len(valores_positivos)
                        else 1.0
                    )

                    if not np.isfinite(limite_cor) or limite_cor <= 0:
                        limite_cor = (
                            float(np.nanmax(valores_validos))
                            if len(valores_validos)
                            else 1.0
                        )

                    fig_heat = px.imshow(
                        heat_pivot,
                        aspect="auto",
                        color_continuous_scale="Blues",
                        zmin=0,
                        zmax=limite_cor,
                        labels={
                            "x": "Mês",
                            "y": "Sala",
                            "color": "Média (L/dia)"
                        },
                        title=(
                            "Consumo médio diário por sala e mês (L/dia)"
                        )
                    )

                    fig_heat.update_layout(
                        coloraxis_colorbar={
                            "title": "L/dia",
                            "tickformat": ",.0f"
                        },
                        margin={
                            "l": 20,
                            "r": 20,
                            "t": 70,
                            "b": 40
                        },
                        height=max(
                            420,
                            min(900, 28 * len(heat_pivot) + 180)
                        )
                    )

                    fig_heat.update_traces(
                        hovertemplate=(
                            "Sala: %{y}<br>"
                            "Mês: %{x}<br>"
                            "Média diária: %{z:,.2f} L/dia"
                            "<extra></extra>"
                        )
                    )

                    st.plotly_chart(
                        fig_heat,
                        width="stretch"
                    )

                    st.caption(
                        "Escala de cores ajustada ao percentil 95 dos "
                        f"valores positivos ({formatar_numero(limite_cor, 1)} "
                        "L/dia). Valores acima desse limite continuam "
                        "disponíveis ao passar o cursor."
                    )

                else:
                    st.info(
                        "Não há dados suficientes para gerar o mapa de calor."
                    )

                # ------------------------------------------------
                # 6. TAXA HORÁRIA EQUIVALENTE
                # ------------------------------------------------

                st.subheader(
                    "6. Taxa horária equivalente (estimativa)"
                )

                st.warning(
                    "Os arquivos contêm consumo diário, não medições "
                    "por hora. A divisão do consumo diário por 24 "
                    "estima uma taxa horária média equivalente, mas "
                    "não permite inferir em quais horas ocorreu "
                    "o consumo."
                )

                diario_sala = (
                    perfil.groupby(
                        ["data_dia", "sala"],
                        as_index=False
                    )
                    .agg(
                        consumo_diario_sala_l=(
                            "consumo_diario",
                            "sum"
                        )
                    )
                )

                media_diaria_edificacao = (
                    perfil.groupby("data_dia", as_index=False)
                    .agg(
                        consumo_diario_edificacao_l=(
                            "consumo_diario",
                            "sum"
                        )
                    )["consumo_diario_edificacao_l"]
                    .mean()
                )

                medias_sala = (
                    diario_sala.groupby(
                        "sala",
                        as_index=False
                    )
                    .agg(
                        consumo_medio_diario_sala_l=(
                            "consumo_diario_sala_l",
                            "mean"
                        ),
                        dias_observados=("data_dia", "nunique")
                    )
                    .sort_values(
                        "consumo_medio_diario_sala_l",
                        ascending=False
                    )
                )

                if not medias_sala.empty:
                    sala_horaria = st.selectbox(
                        "Sala para consultar",
                        options=medias_sala["sala"].tolist(),
                        key="perfil_sala_horaria"
                    )

                    media_diaria_sala = float(
                        medias_sala.loc[
                            medias_sala["sala"] == sala_horaria,
                            "consumo_medio_diario_sala_l"
                        ].iloc[0]
                    )

                    consumo_hora_edificacao = (
                        float(media_diaria_edificacao) / 24
                        if pd.notna(media_diaria_edificacao)
                        else 0.0
                    )

                    consumo_hora_sala = (
                        media_diaria_sala / 24
                    )

                    m1, m2 = st.columns(2)

                    m1.metric(
                        "Edificação — taxa média equivalente",
                        f"{formatar_numero(consumo_hora_edificacao, 2)} L/h"
                    )

                    m2.metric(
                        f"Sala {sala_horaria} — taxa média equivalente",
                        f"{formatar_numero(consumo_hora_sala, 2)} L/h"
                    )

                    tabela_horaria = medias_sala.copy()

                    tabela_horaria[
                        "Consumo médio diário (L/dia)"
                    ] = tabela_horaria[
                        "consumo_medio_diario_sala_l"
                    ]

                    tabela_horaria[
                        "Taxa média equivalente (L/h)"
                    ] = (
                        tabela_horaria[
                            "consumo_medio_diario_sala_l"
                        ] / 24
                    )

                    tabela_horaria = tabela_horaria.rename(
                        columns={
                            "sala": "Sala",
                            "dias_observados": "Dias observados"
                        }
                    )

                    tabela_horaria = tabela_horaria[
                        [
                            "Sala",
                            "Consumo médio diário (L/dia)",
                            "Taxa média equivalente (L/h)",
                            "Dias observados"
                        ]
                    ]

                    with st.expander(
                        "Consultar taxa média equivalente de todas as salas"
                    ):
                        st.dataframe(
                            tabela_horaria,
                            width="stretch",
                            hide_index=True
                        )

                else:
                    st.info(
                        "Não há dados suficientes para calcular "
                        "a taxa horária equivalente."
                    )

                # ------------------------------------------------
                # NOTAS
                # ------------------------------------------------

                st.subheader("Notas de interpretação")

                st.markdown(
                    """
                    - O consumo médio diário por sala é calculado somando
                      os medidores da sala em cada data e, em seguida,
                      calculando a média desses totais diários.
                    - O consumo mensal total corresponde à soma dos
                      registros de consumo diário no mês.
                    - Diferenças entre anos devem ser interpretadas
                      considerando eventuais lacunas ou alterações
                      na cobertura dos medidores.
                    - Leituras negativas foram excluídas da análise
                      descritiva, mas permanecem disponíveis na
                      análise de cobertura.
                    - A taxa em L/h apresentada é uma média equivalente,
                      não um perfil de consumo horário medido.
                    """
                )


# ============================================================
# ANOMALIAS
# ============================================================

with aba_anomalias:
    st.header("🚨 Observações classificadas como anormais")

    anom = df_filtrado[
        df_filtrado["anormal"]
    ].copy()

    if anom.empty:
        st.success(
            "Nenhuma anomalia nos filtros selecionados."
        )

    else:
        a1, a2, a3 = st.columns(3)

        a1.metric(
            "Anomalias",
            len(anom)
        )

        acima = int(
            (
                anom["consumo_real"]
                > anom["consumo_previsto"]
            ).sum()
        )

        abaixo = int(
            (
                anom["consumo_real"]
                < anom["consumo_previsto"]
            ).sum()
        )

        a2.metric(
            "Acima da previsão",
            acima
        )

        a3.metric(
            "Abaixo da previsão",
            abaixo
        )

        st.caption(
            "Um alerta indica desvio em relação à previsão; "
            "não confirma vazamento."
        )

        st.subheader("Direção dos desvios")

        direcao = (
            anom["tipo_desvio"]
            .value_counts()
            .rename_axis("Tipo de desvio")
            .reset_index(name="Quantidade")
        )

        fig_dir = px.bar(
            direcao,
            x="Tipo de desvio",
            y="Quantidade",
            text_auto=True
        )

        st.plotly_chart(
            fig_dir,
            width="stretch"
        )

        st.subheader("Maiores desvios")

        colunas = [
            "sala",
            "codigo_medidor",
            "data_alvo",
            "consumo_real",
            "consumo_previsto",
            "consumo_previsto_ajustado",
            "erro_absoluto",
            "limiar_erro",
            "indice_desvio",
            "tipo_desvio",
            "fonte_limiar"
        ]

        colunas = [
            coluna for coluna in colunas
            if coluna in anom.columns
        ]

        st.dataframe(
            anom.sort_values(
                "indice_desvio",
                ascending=False
            )[colunas].head(100),
            width="stretch",
            hide_index=True
        )

        st.subheader("Anomalias por sala")

        por_sala_anom = (
            anom.groupby("sala")
            .size()
            .reset_index(name="Anomalias")
            .sort_values(
                "Anomalias",
                ascending=False
            )
        )

        st.dataframe(
            por_sala_anom,
            width="stretch",
            hide_index=True
        )


# ============================================================
# ERROS
# ============================================================

with aba_erros:
    st.header("📉 Análise dos erros de previsão")

    if df_filtrado.empty:
        st.warning(
            "Nenhum registro corresponde aos filtros."
        )

    else:
        fig_erro = px.histogram(
            df_filtrado,
            x="erro",
            nbins=50,
            labels={
                "erro": "Erro = consumo real − previsão bruta"
            }
        )

        fig_erro.add_vline(
            x=0,
            line_dash="dash"
        )

        st.plotly_chart(
            fig_erro,
            width="stretch"
        )

        st.subheader("Índice de desvio")

        st.caption(
            "Índice = erro absoluto / limiar efetivo. "
            "Valores acima de 1 indicam anomalia."
        )

        fig_idx = px.histogram(
            df_filtrado,
            x="indice_desvio",
            nbins=50
        )

        fig_idx.add_vline(
            x=1,
            line_dash="dash"
        )

        st.plotly_chart(
            fig_idx,
            width="stretch"
        )

        st.subheader("Maiores erros absolutos")

        n_top = st.slider(
            "Quantidade de registros",
            min_value=5,
            max_value=50,
            value=10,
            key="erros_n_top"
        )

        colunas = [
            "sala",
            "codigo_medidor",
            "data_alvo",
            "consumo_real",
            "consumo_previsto",
            "erro",
            "erro_absoluto",
            "limiar_erro",
            "anormal"
        ]

        colunas = [
            coluna for coluna in colunas
            if coluna in df_filtrado.columns
        ]

        st.dataframe(
            df_filtrado.sort_values(
                "erro_absoluto",
                ascending=False
            ).head(n_top)[colunas],
            width="stretch",
            hide_index=True
        )


# ============================================================
# SÉRIE TEMPORAL
# ============================================================

with aba_series:
    st.header("📈 Análise detalhada de uma série")

    salas_disponiveis = sorted(
        df_filtrado["sala"].dropna().astype(str).unique()
    )

    if not salas_disponiveis:
        st.warning(
            "Não existem salas para os filtros selecionados."
        )

    else:
        sala_escolhida = st.selectbox(
            "Sala para análise",
            salas_disponiveis,
            key="serie_sala"
        )

        df_sala = df_filtrado[
            df_filtrado["sala"] == sala_escolhida
        ].copy()

        medidores_disponiveis = sorted(
            df_sala["codigo_medidor"]
            .dropna()
            .astype(str)
            .unique()
        )

        if not medidores_disponiveis:
            st.warning(
                "Não existem medidores para esta sala."
            )

        else:
            medidor_escolhido = st.selectbox(
                "Medidor para análise",
                medidores_disponiveis,
                key="serie_medidor"
            )

            df_serie = df_sala[
                df_sala["codigo_medidor"] == medidor_escolhido
            ].sort_values("data_alvo").copy()

            fig_serie = go.Figure()

            fig_serie.add_trace(go.Scatter(
                x=df_serie["data_alvo"],
                y=df_serie["consumo_real"],
                mode="lines+markers",
                name="Consumo real"
            ))

            fig_serie.add_trace(go.Scatter(
                x=df_serie["data_alvo"],
                y=df_serie[COLUNA_PREVISAO_GRAFICO],
                mode="lines+markers",
                name="Previsão ajustada"
            ))

            an = df_serie[
                df_serie["anormal"]
            ]

            if not an.empty:
                fig_serie.add_trace(go.Scatter(
                    x=an["data_alvo"],
                    y=an["consumo_real"],
                    mode="markers",
                    name="Observação sinalizada",
                    marker=dict(
                        size=11,
                        symbol="x"
                    )
                ))

            fig_serie.update_layout(
                xaxis_title="Data",
                yaxis_title="Consumo (L)",
                hovermode="x unified"
            )

            st.plotly_chart(
                fig_serie,
                width="stretch"
            )

            st.caption(
                "A linha de previsão é ajustada para exibição; "
                "os alertas continuam usando a previsão bruta."
            )

            st.dataframe(
                df_serie,
                width="stretch",
                hide_index=True
            )


# ============================================================
# DADOS E DOWNLOAD
# ============================================================

with aba_dados:
    st.header("📋 Dados utilizados na análise")

    st.dataframe(
        df_filtrado,
        width="stretch",
        hide_index=True
    )

    csv_download = df_filtrado.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "⬇️ Baixar dados filtrados",
        data=csv_download,
        file_name="dados_filtrados_dashboard.csv",
        mime="text/csv"
    )


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "LSTM + detecção de anomalias por desvio em relação ao "
    "comportamento esperado. Métricas calculadas com previsões brutas."
)
