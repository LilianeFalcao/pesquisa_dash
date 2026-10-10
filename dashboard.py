
import os
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

# Bases brutas usadas para caracterizar os perfis de consumo.
ARQUIVO_BRUTO_2024 = os.path.join(PASTA_DADOS, "dados_medidores_2024.csv")
ARQUIVO_BRUTO_2025 = os.path.join(PASTA_DADOS, "dados_medidores_2025.csv")

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


# ============================================================
# FUNÇÕES
# ============================================================

@st.cache_data
def carregar_dados_brutos(caminho, ano):
    """Carrega e padroniza registros brutos para análise de consumo."""
    bruto = pd.read_csv(caminho, encoding="utf-8-sig", low_memory=False)
    obrigatorias = ["datetime", "day_consumption", "codigo_medidor"]
    faltantes = [c for c in obrigatorias if c not in bruto.columns]
    if faltantes:
        raise ValueError(f"Colunas ausentes em {os.path.basename(caminho)}: {', '.join(faltantes)}")

    bruto["data"] = pd.to_datetime(bruto["datetime"], errors="coerce")
    bruto["consumo_diario"] = pd.to_numeric(bruto["day_consumption"], errors="coerce")
    if "sala_nome_completo" in bruto.columns:
        bruto["sala"] = bruto["sala_nome_completo"].astype("string")
    elif {"sala_bloco", "sala_numero"}.issubset(bruto.columns):
        bruto["sala"] = bruto["sala_bloco"].astype("string") + " - " + bruto["sala_numero"].astype("string")
    else:
        bruto["sala"] = "Sala não identificada"

    bruto["codigo_medidor"] = bruto["codigo_medidor"].astype("string")
    bruto["ano_arquivo"] = ano
    bruto["mes"] = bruto["data"].dt.month
    bruto["mes_nome"] = bruto["mes"].map(MESES)
    bruto["data_dia"] = bruto["data"].dt.date
    # Segunda a sexta = dia útil; sábado e domingo = fim de semana.
    bruto["tipo_dia"] = np.where(bruto["data"].dt.dayofweek < 5, "Dia útil", "Fim de semana")

    # Mantém somente registros com data, sala, medidor e consumo válidos.
    bruto = bruto.dropna(subset=["data", "consumo_diario", "sala", "codigo_medidor"])
    bruto = bruto[bruto["sala"].str.lower().ne("<na>")]
    bruto = bruto[bruto["sala"].str.strip().ne("")]
    # Leituras negativas não são consumo físico válido para os gráficos descritivos.
    bruto = bruto[bruto["consumo_diario"] >= 0].copy()
    return bruto


@st.cache_data
def carregar_csv(caminho):
    return pd.read_csv(caminho, encoding="utf-8-sig")


def converter_bool(serie):
    """Converte corretamente booleanos lidos do CSV."""
    if pd.api.types.is_bool_dtype(serie):
        return serie.fillna(False)

    mapa = {
        "true": True, "false": False,
        "1": True, "0": False,
        "sim": True, "não": False,
        "yes": True, "no": False
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
        "conjunto", "sala", "codigo_medidor", "data_alvo",
        "consumo_real", "consumo_previsto", "anormal",
        "limiar_erro", "erro_absoluto"
    ]

    faltantes = [c for c in obrigatorias if c not in df.columns]

    if faltantes:
        raise ValueError(
            "Colunas ausentes no CSV: " + ", ".join(faltantes)
        )

    df["data_alvo"] = pd.to_datetime(
        df["data_alvo"], errors="coerce"
    )

    for coluna in ["sala", "codigo_medidor", "conjunto"]:
        df[coluna] = df[coluna].astype("string").fillna(
            "Não informado"
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
                df[coluna], errors="coerce"
            )

    df["anormal"] = converter_bool(df["anormal"])

    if "consumo_previsto_ajustado" not in df.columns:
        df["consumo_previsto_ajustado"] = (
            df["consumo_previsto"].clip(lower=0)
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
        df["fonte_limiar"] = "Limiar calibrado em 2024"

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
    Calcula as métricas oficiais usando previsões brutas.
    Viés = previsão bruta - consumo real.
    """
    dados = df.dropna(
        subset=["consumo_real", "consumo_previsto"]
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


# ============================================================
# CARREGAMENTO DO CSV CONSOLIDADO
# ============================================================

st.title("💧 Consumo de Água — Previsão e Detecção de Anomalias")

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
    if "Teste 2024" in opcoes else 0
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
    "Ano", anos, default=anos
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
    .dropna().astype(str).unique()
)

medidores_sel = st.sidebar.multiselect(
    "Medidor (opcional)", medidores
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
    "Tipo de desvio (opcional)", tipos
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

# Gráficos usam previsão ajustada a zero.
# Métricas e classificação usam previsão bruta.
COLUNA_PREVISAO_GRAFICO = "consumo_previsto_ajustado"


# ============================================================
# ABAS
# ============================================================

aba_visao, aba_perfis, aba_anomalias, aba_erros, aba_series, aba_dados = st.tabs([
    "📊 Visão geral",
    "🏢 Perfil de consumo",
    "🚨 Anomalias",
    "📉 Erros",
    "📈 Série temporal",
    "📋 Dados"
])


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
                    COLUNA_PREVISAO_GRAFICO, "mean"
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

        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig_class, width="stretch")

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
                "sala", "codigo_medidor", "data_alvo"
            ],
            labels={
                "consumo_real": "Consumo real (L)",
                COLUNA_PREVISAO_GRAFICO:
                    "Consumo previsto ajustado (L)",
                "color": "Classificação"
            }
        )

        st.plotly_chart(fig_scatter, width="stretch")

        st.info(
            "As previsões exibidas nos gráficos são limitadas a zero. "
            "As métricas oficiais e a classificação de anomalias usam "
            "as previsões brutas."
        )


# ============================================================
# PERFIL DE CONSUMO (DADOS BRUTOS DE 2024 E 2025)
# ============================================================

with aba_perfis:
    st.header("🏢 Caracterização dos perfis de consumo")
    st.markdown(
        "Esta seção utiliza os registros brutos de consumo diário para comparar os períodos "
        "e caracterizar as salas. Os gráficos descritivos não usam as previsões da LSTM. "
        "Leituras negativas e registros sem identificação válida são excluídos desta análise."
    )

    arquivos_brutos = []
    if os.path.exists(ARQUIVO_BRUTO_2024):
        arquivos_brutos.append((ARQUIVO_BRUTO_2024, 2024))
    if os.path.exists(ARQUIVO_BRUTO_2025):
        arquivos_brutos.append((ARQUIVO_BRUTO_2025, 2025))

    if not arquivos_brutos:
        st.warning(
            "Para habilitar esta seção, coloque os arquivos `dados_medidores_2024.csv` e "
            "`dados_medidores_2025.csv` na pasta `dash/` do repositório."
        )
    else:
        partes = []
        erros_carga = []
        for caminho_bruto, ano_bruto in arquivos_brutos:
            try:
                partes.append(carregar_dados_brutos(caminho_bruto, ano_bruto))
            except Exception as exc:
                erros_carga.append(f"{os.path.basename(caminho_bruto)}: {exc}")
        if erros_carga:
            st.warning("Alguns arquivos brutos não puderam ser carregados: " + " | ".join(erros_carga))

        if not partes:
            st.error("Não foi possível carregar os arquivos brutos para a análise de perfil.")
        else:
            perfil = pd.concat(partes, ignore_index=True)
            anos_perfil = sorted(perfil["ano_arquivo"].dropna().unique().tolist())
            col_filtros = st.columns([1, 1, 2])
            with col_filtros[0]:
                anos_perfil_sel = st.multiselect("Ano(s) para analisar", anos_perfil, default=anos_perfil, key="perfil_anos")
            perfil = perfil[perfil["ano_arquivo"].isin(anos_perfil_sel)].copy()
            with col_filtros[1]:
                meses_perfil_sel = st.multiselect("Meses", list(range(1, 13)), default=list(range(1, 13)), format_func=lambda x: MESES[x], key="perfil_meses")
            perfil = perfil[perfil["mes"].isin(meses_perfil_sel)].copy()
            salas_perfil = sorted(perfil["sala"].dropna().unique().tolist())
            with col_filtros[2]:
                salas_perfil_sel = st.multiselect("Salas (opcional)", salas_perfil, key="perfil_salas")
            if salas_perfil_sel:
                perfil = perfil[perfil["sala"].isin(salas_perfil_sel)].copy()

            if perfil.empty:
                st.info("Não há registros para os filtros selecionados.")
            else:
                mensal = (
                    perfil.assign(mes_data=perfil["data"].dt.to_period("M").dt.to_timestamp())
                    .groupby(["ano_arquivo", "mes_data"], as_index=False)
                    .agg(consumo_total_l=("consumo_diario", "sum"),
                         consumo_medio_registro_l=("consumo_diario", "mean"),
                         dias_registrados=("data_dia", "nunique"),
                         medidores=("codigo_medidor", "nunique"))
                )
                # Indicador diário agregado: total mensal dividido pelos dias do calendário cobertos no mês.
                mensal["consumo_medio_diario_l"] = mensal["consumo_total_l"] / mensal["mes_data"].dt.days_in_month
                total_l = perfil["consumo_diario"].sum()
                media_registro = perfil["consumo_diario"].mean()
                mediana_registro = perfil["consumo_diario"].median()
                salas_n = perfil["sala"].nunique()
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Consumo acumulado nos filtros", f"{formatar_numero(total_l, 1)} L")
                k2.metric("Média por registro", f"{formatar_numero(media_registro, 2)} L")
                k3.metric("Mediana por registro", f"{formatar_numero(mediana_registro, 2)} L")
                k4.metric("Salas analisadas", f"{salas_n:,}".replace(",", "."))

                st.subheader("1. Comparação entre dias úteis e finais de semana")
                # Agrega primeiro todos os medidores por data, para comparar o total diário da edificação.
                diario_completo = (
                    perfil.groupby(["data_dia", "tipo_dia"], as_index=False)
                    .agg(consumo_total_dia_l=("consumo_diario", "sum"))
                )
                comparacao_dias = (
                    diario_completo.groupby("tipo_dia", as_index=False)
                    .agg(media_consumo_diario_l=("consumo_total_dia_l", "mean"),
                         mediana_consumo_diario_l=("consumo_total_dia_l", "median"),
                         dias_observados=("data_dia", "nunique"))
                )
                ordem_tipos = ["Dia útil", "Fim de semana"]
                comparacao_dias["tipo_dia"] = pd.Categorical(comparacao_dias["tipo_dia"], categories=ordem_tipos, ordered=True)
                comparacao_dias = comparacao_dias.sort_values("tipo_dia")
                fig_dias = px.bar(
                    comparacao_dias, x="tipo_dia", y="media_consumo_diario_l",
                    error_y=None,
                    labels={"tipo_dia": "Tipo de dia", "media_consumo_diario_l": "Consumo médio diário total (L)"},
                    title="Média do consumo diário total: dias úteis × finais de semana",
                    hover_data={"mediana_consumo_diario_l": ":.2f", "dias_observados": True}
                )
                st.plotly_chart(fig_dias, width="stretch")
                st.caption("Dias úteis são considerados de segunda a sexta-feira; finais de semana, sábado e domingo. Feriados não são classificados separadamente.")

                st.subheader("2. Evolução do consumo mensal total")
                fig_mensal = px.line(
                    mensal, x="mes_data", y="consumo_total_l", color="ano_arquivo", markers=True,
                    labels={"mes_data": "Mês", "consumo_total_l": "Consumo total (L)", "ano_arquivo": "Ano"},
                    title="Consumo total registrado por mês"
                )
                fig_mensal.update_layout(hovermode="x unified")
                st.plotly_chart(fig_mensal, width="stretch")

                st.subheader("3. Comparação entre anos por mês do calendário")
                mensal["mes_numero"] = mensal["mes_data"].dt.month
                comparacao = mensal.groupby(["ano_arquivo", "mes_numero"], as_index=False).agg(consumo_total_l=("consumo_total_l", "sum"))
                comparacao["mes_nome"] = comparacao["mes_numero"].map(MESES)
                fig_comp = px.bar(
                    comparacao, x="mes_nome", y="consumo_total_l", color="ano_arquivo", barmode="group",
                    category_orders={"mes_nome": list(MESES.values())},
                    labels={"mes_nome": "Mês", "consumo_total_l": "Consumo total (L)", "ano_arquivo": "Ano"},
                    title="Comparação do consumo mensal entre anos"
                )
                st.plotly_chart(fig_comp, width="stretch")
                st.caption("Compare meses equivalentes. Se um ano tiver registros incompletos, a diferença pode refletir também a cobertura dos dados.")

                st.subheader("4. Consumo médio diário por sala")
                por_sala = (
                    perfil.groupby(["sala", "ano_arquivo"], as_index=False)
                    .agg(consumo_medio_diario_l=("consumo_diario", "mean"),
                         consumo_mediano_diario_l=("consumo_diario", "median"),
                         observacoes=("consumo_diario", "size"),
                         medidores=("codigo_medidor", "nunique"))
                )
                col_top, col_escala = st.columns([2, 1])
                with col_top:
                    top_n = st.slider("Quantidade de salas no gráfico", min_value=5, max_value=40, value=15, key="perfil_top_n")
                with col_escala:
                    escala_salas = st.selectbox(
                        "Escala do eixo de consumo",
                        options=["Linear", "Logarítmica"],
                        index=0,
                        key="perfil_escala_salas",
                        help="A escala logarítmica ajuda a comparar salas quando há diferenças muito grandes entre os valores."
                    )
                ordenar_ano = anos_perfil_sel[-1] if anos_perfil_sel else int(perfil["ano_arquivo"].max())
                ranking = por_sala[por_sala["ano_arquivo"] == ordenar_ano].sort_values("consumo_medio_diario_l", ascending=False).head(top_n)
                if not ranking.empty:
                    ranking_plot = ranking.sort_values("consumo_medio_diario_l").copy()
                    if escala_salas == "Logarítmica":
                        ranking_plot = ranking_plot[ranking_plot["consumo_medio_diario_l"] > 0]
                    if not ranking_plot.empty:
                        fig_salas = px.bar(
                            ranking_plot, x="consumo_medio_diario_l", y="sala", orientation="h",
                            hover_data=["observacoes", "medidores"],
                            labels={"consumo_medio_diario_l": "Média por registro (L)", "sala": "Sala"},
                            title=f"{len(ranking_plot)} salas com maior média por registro — {ordenar_ano}"
                        )
                        if escala_salas == "Logarítmica":
                            fig_salas.update_xaxes(type="log", title="Média por registro (L) — escala logarítmica")
                            st.caption("A escala logarítmica facilita a visualização de diferenças entre valores muito distantes. Valores iguais a zero não aparecem nessa escala; os dados originais não são alterados.")
                        st.plotly_chart(fig_salas, width="stretch")
                    else:
                        st.info("Não há valores positivos para exibir na escala logarítmica.")
                st.dataframe(por_sala.sort_values(["ano_arquivo", "consumo_medio_diario_l"], ascending=[True, False]), width="stretch", hide_index=True)

                st.subheader("5. Mapa de calor: sala × mês")
                perfil_heat = perfil.copy()
                perfil_heat["mes_nome"] = perfil_heat["mes"].map(MESES)
                heat = perfil_heat.groupby(["sala", "mes_nome"], as_index=False).agg(media_l=("consumo_diario", "mean"))
                heat_pivot = heat.pivot(index="sala", columns="mes_nome", values="media_l")
                heat_pivot = heat_pivot.reindex(columns=[m for m in MESES.values() if m in heat_pivot.columns])
                salas_heat = por_sala.sort_values("consumo_medio_diario_l", ascending=False)["sala"].drop_duplicates().head(30).tolist()
                heat_pivot = heat_pivot.loc[heat_pivot.index.intersection(salas_heat)]
                if not heat_pivot.empty:
                    fig_heat = px.imshow(
                        heat_pivot, aspect="auto", color_continuous_scale="Blues",
                        labels={"x": "Mês", "y": "Sala", "color": "Média (L)"},
                        title="Média de consumo por registro, sala e mês"
                    )
                    st.plotly_chart(fig_heat, width="stretch")

                st.subheader("Notas de interpretação")
                st.markdown(
                    "- A média por sala é calculada a partir dos registros de consumo diário disponíveis para a sala no período filtrado.\n"
                    "- O consumo mensal total corresponde à soma dos registros de consumo diário no mês.\n"
                    "- Os gráficos descrevem os dados disponíveis; diferenças entre anos devem ser interpretadas considerando eventuais lacunas ou alterações na cobertura dos medidores.\n"
                    "- Leituras negativas foram excluídas da análise descritiva, pois não representam consumo físico positivo."
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

        a1.metric("Anomalias", len(anom))

        acima = int(
            (anom["consumo_real"] > anom["consumo_previsto"]).sum()
        )
        abaixo = int(
            (anom["consumo_real"] < anom["consumo_previsto"]).sum()
        )

        a2.metric("Acima da previsão", acima)
        a3.metric("Abaixo da previsão", abaixo)

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
        st.plotly_chart(fig_dir, width="stretch")

        st.subheader("Maiores desvios")

        colunas = [
            "sala", "codigo_medidor", "data_alvo",
            "consumo_real", "consumo_previsto",
            "consumo_previsto_ajustado",
            "erro_absoluto", "limiar_erro",
            "indice_desvio", "tipo_desvio", "fonte_limiar"
        ]

        colunas = [
            c for c in colunas if c in anom.columns
        ]

        st.dataframe(
            anom.sort_values(
                "indice_desvio", ascending=False
            )[colunas].head(100),
            width="stretch",
            hide_index=True
        )

        st.subheader("Anomalias por sala")

        por_sala = (
            anom.groupby("sala")
            .size()
            .reset_index(name="Anomalias")
            .sort_values("Anomalias", ascending=False)
        )

        st.dataframe(
            por_sala,
            width="stretch",
            hide_index=True
        )


# ============================================================
# ERROS
# ============================================================

with aba_erros:
    st.header("📉 Análise dos erros de previsão")

    if df_filtrado.empty:
        st.warning("Nenhum registro corresponde aos filtros.")
    else:
        fig_erro = px.histogram(
            df_filtrado,
            x="erro",
            nbins=50,
            labels={
                "erro": "Erro = consumo real − previsão bruta"
            }
        )

        fig_erro.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig_erro, width="stretch")

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
        fig_idx.add_vline(x=1, line_dash="dash")

        st.plotly_chart(fig_idx, width="stretch")

        st.subheader("Maiores erros absolutos")

        n_top = st.slider(
            "Quantidade de registros",
            min_value=5,
            max_value=50,
            value=10
        )

        colunas = [
            "sala", "codigo_medidor", "data_alvo",
            "consumo_real", "consumo_previsto",
            "erro", "erro_absoluto", "limiar_erro", "anormal"
        ]

        colunas = [
            c for c in colunas if c in df_filtrado.columns
        ]

        st.dataframe(
            df_filtrado.sort_values(
                "erro_absoluto", ascending=False
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
            salas_disponiveis
        )

        df_sala = df_filtrado[
            df_filtrado["sala"] == sala_escolhida
        ].copy()

        medidores_disponiveis = sorted(
            df_sala["codigo_medidor"]
            .dropna().astype(str).unique()
        )

        if not medidores_disponiveis:
            st.warning("Não existem medidores para esta sala.")
        else:
            medidor_escolhido = st.selectbox(
                "Medidor para análise",
                medidores_disponiveis
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

            an = df_serie[df_serie["anormal"]]

            if not an.empty:
                fig_serie.add_trace(go.Scatter(
                    x=an["data_alvo"],
                    y=an["consumo_real"],
                    mode="markers",
                    name="Observação sinalizada",
                    marker=dict(size=11, symbol="x")
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

st.divider()

st.caption(
    "LSTM + detecção de anomalias por desvio em relação ao "
    "comportamento esperado. Métricas calculadas com previsões brutas."
)